__author__ = 'jparedes'

from sklearn import metrics
from decisions import Decisions
import numpy as np
from itertools import chain


class Evaluation:
    def __init__(self, model="a", classes_premises="b", priority_classes="c"):
        self.modelo = model
        self.classes_premises = classes_premises
        self.complementary_parameters = priority_classes

    @staticmethod
    def dummies2int(binary_classes):
        return [i.index(1) + 1 for i in binary_classes.tolist()]

    @staticmethod
    def metricas(reference, estimation, premisas_das_clases):
        num_classes = len(premisas_das_clases)

        premisas = list(chain(*premisas_das_clases))
        num_regras = len(premisas)  # list or tuple
        total_rule_length = len(list(chain(*premisas)))

        tamano_medio_das_regras = (1. * total_rule_length) / num_regras
        metrica_premisas = [num_regras, total_rule_length, tamano_medio_das_regras]

        'Calculate Area Under the Curve in a problem with 2 classes'
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
        return [metrics.accuracy_score(reference, estimation), i1, i2, i3, metrica_premisas]

    def eval_train(self, reference_bin, estimation_bin):
        reference = self.dummies2int(reference_bin)
        estimation = self.dummies2int(estimation_bin)
        if self.classes_premises == "b":
            self.classes_premises = [[(i, 0)] for i in range(reference_bin.shape[1])]
        return self.metricas(reference, estimation, self.classes_premises)

    def eval_test(self, cbin_test, u_test, t_norm):
        num_classes = cbin_test.shape[1]
        ybin_est = np.zeros((u_test.shape[0], num_classes))
        for i in range(num_classes):
            a = self.eval_model(self.modelo[i], u_test, t_norm)
            ybin_est[:, [i]] = a
        aux = Decisions(ybin_est, self.complementary_parameters)
        cbin_test_estimation = aux.dec_max_pert()

        reference = self.dummies2int(cbin_test)
        estimation = self.dummies2int(cbin_test_estimation)
        return self.metricas(reference, estimation, self.classes_premises)

    def eval_model(self, modelo, u, tnorm):  # log regression con prod se friega bastante
        # modelo: [referencias_premisas, weights, name_model]
        name_model = modelo[2]
        ind_u = modelo[0]

        new_u = self.build_u_test(u, ind_u, tnorm)

        if name_model == 'MQR' or name_model == 'intMQR' or name_model == 'lin_regre':
            weights = modelo[1]
            return np.dot(new_u, weights)

        elif name_model == 'max':
            aux = new_u.max(axis=1)  # np.max(new_u, axis=1)
            return aux.reshape(len(aux), 1)

        else:  # 'logistic regression'
            model = modelo[1]
            aux = model.predict(new_u)
            return aux.reshape(len(aux), 1)

    @staticmethod
    def build_u_test(u, indexes, tnorm):
        num_col = len(indexes)
        new_u = np.zeros((u.shape[0], num_col))
        for i in range(num_col):
            temp = u[:, indexes[i]]
            if tnorm == 'prod':
                new_u[:, i] = np.prod(temp, axis=1)
            else:  # 'min'
                new_u[:, i] = np.min(temp, axis=1)
        return new_u


def main():
    print ("Welcome to Module <Evaluation>")


if __name__ == '__main__':
    main()