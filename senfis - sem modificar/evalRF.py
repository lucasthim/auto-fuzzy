
from numpy import dot, hstack, tile, prod, zeros, min
from itertools import compress, chain
from sklearn import metrics
from autoFIS.autoFIS.decisions import Decisions


def eval_classifiers(classifiers, ux_Test, tnorm, freq_classes):
    # num_classes = len(classifier)
    # for i in range(num_classes):
    #     a = 5
    bin_output_classifiers = []
    for classifier in classifiers:
        output_classifier = []
        for infmodel_i in classifier:  # by class
            output_classifier.append(eval_model_autofis(infmodel_i, ux_Test, tnorm))
        aux = hstack(output_classifier)  # salida del clasificador aun no binarizada, no normalizada
        aux_norm = normalizar(aux)  # normalizando la salida del clasificador
        bin_output_classifiers.append(binarizar(aux_norm, freq_classes))  # salida del clasificador binarizada
    return bin_output_classifiers


def eval_classifier_one(infmodel, ux_Test, tnorm):
    output_classifier = []
    for infmodel_i in infmodel:  # by class
        output_classifier.append(eval_model_autofis(infmodel_i, ux_Test, tnorm))
    aux = hstack(output_classifier)  # salida del clasificador aun no binarizada, no normalizada
    aux_norm = normalizar(aux)  # normalizando la salida del clasificador
    return aux_norm


def normalizar(aux):
    sum_fila = aux.sum(1)
    c = tile(sum_fila, (aux.shape[1], 1))
    aux_norm = aux / c.T
    return aux_norm


def binarizar(m, freq_classes):
    """
    This function decide to which class belong each instance
    :param m: estimation
    :param freq_classes: [0.2 0.3 0.5]
    :return: Binary classification

    self.aggregation_values [0.80 0.15 0.05
                             0.03 0.90 0.07
                             0.20 0.40 0.40
                             0.20 0.10 0.70]
    self.decision_parameters:
                     freq_classes: [0.33 0.33 0.33]
    Output:
                    [1, 0, 0]
                    [0, 1, 0]
                    [0, 1, 0]
                    [0, 0, 1]
    """
    repeated_max_values_rows = tile(m.max(1), (m.shape[1], 1))  # m.max(1) maximo de cada fila
    out = 1. * (m == repeated_max_values_rows.T)  # First binary decision (possibles ties)

    d_row_max = out.sum(axis=1) > 1  # Instances with compatibility in more than 1 class (tie)
    index_ties = list(compress(range(out.shape[0]), d_row_max.tolist()))  # index de las instancias com empate

    classes_number = len(freq_classes)
    for j in index_ties:
        tiebreaker = classes_number * [0]
        freq_classes_in_tie = [freq_classes[i] * out[j, i] for i in range(classes_number)]
        tiebreaker[freq_classes_in_tie.index(max(freq_classes_in_tie))] = 1  # Assigning to the class with more patterns
        out[j, :] = tiebreaker

    return out


def eval_model_autofis(modelo, u, tnorm):
    # modelo: [premisas, weights, name_model]
    name_model = modelo[2]
    ind_u = modelo[0]

    new_u = calculation_premises(ind_u, u, tnorm)

    if name_model == 'MQR' or name_model == 'intMQR' or name_model == 'lin_regre':
        weights = modelo[1]
        return dot(new_u, weights)
    elif name_model == 'max':
        aux = new_u.max(axis=1)  # np.max(new_u, axis=1)
        return aux.reshape(len(aux), 1)
    else:  # 'logistic regression'
        model = modelo[1]
        aux = model.predict(new_u)
        return aux.reshape(len(aux), 1)


def calculation_premises(indexes, ux, t_norm):
    number_rows = ux.shape[0]
    new_ux_prev = []
    for i in range(len(indexes)):
        # Building the new premise
        temp = ux[:, indexes[i]]
        if t_norm == 'prod':
            temp2 = temp.prod(axis=1)
        else:  # 'min'
            temp2 = temp.min(axis=1)
        new_ux_prev.append(temp2.reshape(number_rows, 1))
    new_ux = hstack(new_ux_prev)
    return new_ux


def dummies2int(binary_classes):
    return [i.index(1) + 1 for i in binary_classes.tolist()]


def metricas_rules_premises(premisas_das_clases):

    premisas = list(chain(*premisas_das_clases))
    num_regras = len(premisas)  # list or tuple
    total_rule_length = len(list(chain(*premisas)))

    tamano_medio_das_regras = (1. * total_rule_length) / num_regras
    metrica_premisas = [num_regras, total_rule_length, tamano_medio_das_regras]

    return metrica_premisas


def metricas(reference, estimation, num_classes):

    # Calculate Area Under the Curve in a problem with 2 classes

    try:
        fpr, tpr, thresholds = metrics.roc_curve(reference, estimation, pos_label=num_classes)
        i1 = metrics.auc(fpr, tpr)
        i2 = metrics.confusion_matrix(reference, estimation)
        i3 = metrics.recall_score(reference, estimation, average=None)
    except ValueError:  # it can be improved but at this moment i do not know
            i1, i2, i3 = 0, 0, 0
        # fpr, tpr, thresholds = metrics.roc_curve(reference, estimation, pos_label=num_classes)
        # i1 = metrics.auc(fpr, tpr)
        # i2 = metrics.confusion_matrix(reference, estimation)
        # i3 = metrics.recall_score(reference, estimation, average=None)

    return [100 * metrics.accuracy_score(reference, estimation), i1, i2, i3]


def eval_metrics(reference_bin, estimation_bin):
    num_classes = reference_bin.shape[1]
    reference = dummies2int(reference_bin)
    estimation = dummies2int(estimation_bin)

    return metricas(reference, estimation, num_classes)


def eval_test(modelo, cbin_test, u_test, t_norm, freq_class):
    num_classes = cbin_test.shape[1]
    ybin_est = zeros((u_test.shape[0], num_classes))
    for i in range(num_classes):
        a = eval_model(modelo[i], u_test, t_norm)
        ybin_est[:, [i]] = a
    aux = Decisions(ybin_est, freq_class)
    cbin_test_estimation = aux.dec_max_pert()

    reference = dummies2int(cbin_test)
    estimation = dummies2int(cbin_test_estimation)

    return metricas(reference, estimation, num_classes)


def eval_model(self, modelo, u, tnorm):  # log regression con prod se friega bastante
    # modelo: [referencias_premisas, weights, name_model]
    name_model = modelo[2]
    ind_u = modelo[0]

    new_u = self.build_u_test(u, ind_u, tnorm)

    if name_model == 'MQR' or name_model == 'intMQR' or name_model == 'lin_regre':
        weights = modelo[1]

        return dot(new_u, weights)

    elif name_model == 'max':
        aux = new_u.max(axis=1)  # np.max(new_u, axis=1)
        return aux.reshape(len(aux), 1)

    else:  # 'logistic regression'
        model = modelo[1]
        aux = model.predict(new_u)

        return aux.reshape(len(aux), 1)


def build_u_test(u, indexes, tnorm):
    num_col = len(indexes)
    new_u = zeros((u.shape[0], num_col))
    for i in range(num_col):
        temp = u[:, indexes[i]]
        if tnorm == 'prod':
            new_u[:, i] = prod(temp, axis=1)
        else:  # 'min'
            new_u[:, i] = min(temp, axis=1)

    return new_u

