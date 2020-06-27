import autoFIS.autoFIS.utils_autofis as toolfis
from autoFIS.autoFIS.decisions import Decisions
from autoFIS.autoFIS.evaluation import Evaluation
from numpy import hstack, array, mean
from evalRF import eval_metrics, eval_classifier_one, normalizar, metricas_rules_premises
import os
import timeit
from itertools import chain
from collections import defaultdict
from parameters_init import GlobalParameter
import _thread
import _threading_local


def random_fis_one_cv(zipFilePath, file_train, file_test, parameters_classifiers, cv_i, clf_n, folder__tree_output,
                      time_ini):

    # General parameters
    p_method_agg = GlobalParameter()
    method_aggreg_classf = p_method_agg.method_aggregation
    cv_i += 1
    print ('Clf = %i, CV = %i' % (clf_n, cv_i))
    successful_classifiers = 0
    classifiers = []
    outputs_tree_train_bin = []
    outputs_tree_test_bin = []
    outputs_tree_train = []
    outputs_tree_test = []
    partial_metrics_rules = []
    container_ac_train = []
    container_ac_test = []
    parameters = [parameters_classifiers[0]] + parameters_classifiers[1]
    t_norm = parameters[3]
    max_size_of_premise = parameters[5]
    association_method = parameters[11]
    aggregation_method = parameters[12]
    aux_blb = []

    # Gathering parameters
    # Formulation parameters:
    par_area, par_over, par_pcd = toolfis.get_formulation_parameters(parameters)

    # 1. Lecture & Fuzzification
    out1 = toolfis.lecture_fuz_one_cv(zipFilePath, file_train, file_test, parameters, cv_i)
    ux_train, cbin_train = out1[0]
    ux_test, cbin_test = out1[1]
    sizes_attributes, premises_by_attribute, ref_attributes, premises_contain_negation = out1[2]
    freq_classes = out1[3]

    pars = [max_size_of_premise, t_norm, par_area, par_over, par_pcd,
            association_method, aggregation_method, freq_classes]

# ================================= GENERATION CLASSIFIERS =========================================================
    p_blb = GlobalParameter()
    for i in range(parameters[-4]):
        ux_train_blb, new_y_bin_blb, index_oob = toolfis.data_blb(ux_train, cbin_train)
        classifiers_blb = []
        container_ac_test_blb = []
        container_ac_obb_blb = []
        n_blb = list(range(p_blb.blb))
        classifiers_colector = {}
        for blb_i in n_blb:
            new_data_blb, genesis_data_blb = toolfis.create_data(ref_attributes, sizes_attributes,
                                                                 premises_by_attribute,
                                                                 premises_contain_negation,
                                                                 ux_train_blb, new_y_bin_blb)
            exit_flag_blb, out_model_blb = toolfis.inference_fuzzy(new_data_blb, pars, ensemble='RandomFIS')
            #  out_model = [premises_weights_names, train_bin_prediction, estimation_classes(u_estimation)]

            if exit_flag_blb:
                # successful_classifiers += 1
                #  Transformation premises relative2absolute:
                converter_blb = dict(zip(range(len(genesis_data_blb[0])), genesis_data_blb[0]))
                absolute_model_blb = []
                end_premises_classes_blb = []
                for j in out_model_blb[0]:  # by Class
                    relative_premises_blb = j[0]
                    absolute_premises_blb = toolfis.relative2absolute(relative_premises_blb, converter_blb)
                    end_premises_classes_blb.append(absolute_premises_blb)  # For estimation metrics rules and premises
                    absolute_model_blb.append([absolute_premises_blb, j[1], j[2]])  # premises absolutes, Weights, name_method
                classifiers_blb.append(absolute_model_blb)

                # outputs_tree_test_blb = eval_classifier_one(absolute_model_blb, ux_test, t_norm)  # Out U by class test
                # outputs_tree_test_bin_blb = decision_class(outputs_tree_test_blb, freq_classes)
                # container_ac_test_blb.append(round(eval_metrics(cbin_test, outputs_tree_test_bin_blb)[0], 2))
                #
                # #  Obb
                # outputs_tree_obb_blb = eval_classifier_one(absolute_model_blb, ux_train[index_oob, :], t_norm)
                # outputs_tree_obb_bin_blb = decision_class(outputs_tree_obb_blb, freq_classes)
                # container_ac_obb_blb.append(round(eval_metrics(outputs_tree_obb_bin_blb, cbin_train[index_oob, :])[0], 2))
                # rank_res = dict(zip(container_ac_obb_blb, container_ac_test_blb))
                # rank_classifiers(container_ac_obb_blb[-1], blb_i, classifiers_colector)

        num_classes = cbin_train.shape[1]
        indexes_premises_byclass = []
        for ci in range(num_classes):  # debo de colocar aqui el numero de clases
            container_aux = []
            for j in classifiers_blb:
                container_aux.append(j[ci][0])

            list_premises_container = list(chain(*container_aux))
            unique_indexes = list(set(list_premises_container))
            unique_premises = toolfis.calculation_premises(unique_indexes, ux_train_blb, t_norm)
            indexes_premises_byclass.append([unique_indexes, unique_premises])

        success, out_model = toolfis.classifiers_aggregation(indexes_premises_byclass, new_y_bin_blb, 'MQR',
                                                             freq_classes)

        if success:
            successful_classifiers += 1
            absolute_model = []
            end_premises_classes = []
            for j in out_model[0]:  # by Class
                absolute_premises = j[0]
                end_premises_classes.append(absolute_premises)  # For estimation metrics rules and premises
                absolute_model.append([absolute_premises, j[1], j[2]])  # premises absolutes, Weights, name_method
            classifiers.append(absolute_model)
            # Metrics Train
            container_ac_train.append(eval_metrics(new_y_bin_blb, out_model[1])[0])

            outputs_tree_train.append(eval_classifier_one(absolute_model, ux_train, t_norm))  # Out U by class train
            outputs_tree_train_bin.append(decision_class(outputs_tree_train[-1], freq_classes))

            # Metrics Test
            outputs_tree_test.append(eval_classifier_one(absolute_model, ux_test, t_norm))  # Out U by class test
            outputs_tree_test_bin.append(decision_class(outputs_tree_test[-1], freq_classes))
            container_ac_test.append(round(eval_metrics(cbin_test, outputs_tree_test_bin[-1])[0], 2))

            aux_metrics = metricas_rules_premises(end_premises_classes)
            partial_metrics_rules.append(hstack(aux_metrics))
    # with open('ac_agregation_blb.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile)
    #     wr.writerow(container_ac_test)

    if not classifiers:
        return ["Any of %i classifiers was successful" % successful_classifiers], [0], cv_i

    # folder__tree_output_test = os.path.join(folder__tree_output, 'cv_test_%i' % cv_i)
    # with open(folder__tree_output_test, 'w') as f:
    #     pickle.dump(outputs_tree_test, f)
    #
    # folder__tree_output_test = os.path.join(folder__tree_output, 'c_bin_%i' % cv_i)
    # with open(folder__tree_output_test, 'w') as f:
    #     pickle.dump(cbin_test, f)
    #
    # if cv_i == 1:
    #     folder__tree_output_test = os.path.join(folder__tree_output, 'freq')
    #     with open(folder__tree_output_test, 'w') as f:
    #         pickle.dump(freq_classes, f)

    if method_aggreg_classf == 1:

        #  ============================== AVERAGE AGGREGATION ================================================

        #  Evaluation Train
        u_by_class = estimation_average(outputs_tree_train)
        train_decision = decision_class(u_by_class, freq_classes)
        metrics_train = eval_metrics(cbin_train, train_decision)

        #  Evaluation Test_average
        u_by_class = estimation_average(outputs_tree_test)
        test_decision = decision_class(u_by_class, freq_classes)
        metrics_test = eval_metrics(cbin_test, test_decision)

        #  Evaluation rules
        aux_metrics_rules = sum(partial_metrics_rules, 0)
        metrics_rules = aux_metrics_rules

        metrics = [1, [metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1], metrics_rules],
                   successful_classifiers]
        results = [successful_classifiers, metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1],
                   metrics_rules[0], metrics_rules[1]]

        report = toolfis.template_results('', results)

        return report, metrics, cv_i

    elif method_aggreg_classf == 2:

        #  ============================== WEIGHT AGGREGATION ================================================

        weight_by_classifier = estimation_weight(outputs_tree_train, cbin_train)  # Class train binario

        # Evaluation Train
        output_tree_train_aggregated = aggregation_classifiers_by_weight(weight_by_classifier, outputs_tree_train)
        train_decision = decision_class(output_tree_train_aggregated, freq_classes)
        metrics_train = eval_metrics(cbin_train, train_decision)

        # Evaluation Test
        output_tree_test_aggregated = aggregation_classifiers_by_weight(weight_by_classifier, outputs_tree_test)
        test_decision = decision_class(output_tree_test_aggregated, freq_classes)
        metrics_test = eval_metrics(cbin_test, test_decision)

        #  Evaluation rules
        aux_metrics_rules = sum(partial_metrics_rules, 0)
        constant = len(partial_metrics_rules)
        metrics_rules = aux_metrics_rules / constant

        metrics = [1, [metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1], metrics_rules],
                   successful_classifiers]
        results = [successful_classifiers, metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1],
                   metrics_rules[0], metrics_rules[1]]

        report = toolfis.template_results('', results)
        return report, metrics, cv_i

    elif method_aggreg_classf == 3:

        #  ============================== WEIGHT_BY_CLASS AGGREGATION ================================================

        weight_class_by_classifier = estimation_weight_by_class(outputs_tree_train, cbin_train)  # Class train binario

        # Evaluation Train
        output_tree_train_aggregated = aggregation_classifiers_by_weight_class(weight_class_by_classifier,
                                                                               outputs_tree_train)
        train_decision = decision_class(output_tree_train_aggregated, freq_classes)
        metrics_train = eval_metrics(cbin_train, train_decision)

        # Evaluation Test
        output_tree_test_aggregated = aggregation_classifiers_by_weight_class(weight_class_by_classifier,
                                                                              outputs_tree_test)
        test_decision = decision_class(output_tree_test_aggregated, freq_classes)
        metrics_test = eval_metrics(cbin_test, test_decision)

        #  Evaluation rules
        aux_metrics_rules = sum(partial_metrics_rules, 0)
        constant = len(partial_metrics_rules)
        metrics_rules = aux_metrics_rules / constant

        metrics = [1, [metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1], metrics_rules],
                   successful_classifiers]
        results = [successful_classifiers, metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1],
                   metrics_rules[0], metrics_rules[1]]

        report = toolfis.template_results('', results)
        return report, metrics, cv_i

    elif method_aggreg_classf == 4:

        #  ============================== PACKING ONE MODEL V.1 ================================================

        #  Evaluation Test_average
        u_by_class_ave = estimation_average(outputs_tree_test)
        test_decision_ave = decision_class(u_by_class_ave, freq_classes)
        metrics_test_ave = round(eval_metrics(cbin_test, test_decision_ave)[0], 2)

        voting_bin = voting(outputs_tree_test_bin)
        vot_bin = decision_class(voting_bin, freq_classes)
        metrics_test_vot = round(eval_metrics(cbin_test, vot_bin)[0], 2)

        #  Evaluation rules
        aux_metrics_rules_ave = sum(partial_metrics_rules, 0)
        metrics_rules_ave = aux_metrics_rules_ave[0]
        aux_t0 = timeit.default_timer()
        time_finished = aux_t0 - time_ini

        folder__tree_output_test = os.path.join(folder__tree_output, 'ac_teste_av.txt')
        with open(folder__tree_output_test, 'a') as f:
            f.write(str(metrics_test_ave)+'\n')

        folder__tree_output_test = os.path.join(folder__tree_output, 'ac_teste_vot.txt')
        with open(folder__tree_output_test, 'a') as f:
            f.write(str(metrics_test_vot) + '\n')

        folder__tree_output_test = os.path.join(folder__tree_output, 'regras_teste_10CV.txt')
        with open(folder__tree_output_test, 'a') as f:
            f.write(str(metrics_rules_ave) + '\n')

        if cv_i == 10:
            folder__tree_output_test = os.path.join(folder__tree_output, 'time_teste_10CV.txt')
            with open(folder__tree_output_test, 'a') as f:
                f.write(str(time_finished) + '\n')

        num_classes = cbin_train.shape[1]
        indexes_premises_byclass = []
        for i in range(num_classes):  # debo de colocar aqui el numero de clases
            container_aux = []
            for j in classifiers:
                container_aux.append(j[i][0])

            list_premises_container = list(chain(*container_aux))
            unique_indexes = list(set(list_premises_container))
            unique_premises = toolfis.calculation_premises(unique_indexes, ux_train, t_norm)
            indexes_premises_byclass.append([unique_indexes, unique_premises])

        exito, output_collective = toolfis.classifiers_aggregation(indexes_premises_byclass, cbin_train, 'MQR'
                                                                   , freq_classes)

        if exito:
            premises_weights_names = output_collective[0]
            estimation_classes = output_collective[1]

            final_premises_classes = []
            for i in range(len(premises_weights_names)):  # x cada clase
                final_premises_classes.append(premises_weights_names[i][0])

            F6 = Evaluation(premises_weights_names, final_premises_classes, freq_classes)
            metrics_train = F6.eval_train(cbin_train, estimation_classes)
            metrics_test = F6.eval_test(cbin_test, ux_test, t_norm)
            metrics_rules = metrics_test[4]

            metrics = [1, [metrics_train[0]*100, metrics_test[0]*100, metrics_train[1], metrics_test[1], metrics_rules],
                       successful_classifiers]
            results = [successful_classifiers, metrics_train[0]*100, metrics_test[0]*100, metrics_train[1], metrics_test[1],
                       metrics_rules[0], metrics_rules[1]]

            report = toolfis.template_results('', results)
            return report, metrics, cv_i

    elif method_aggreg_classf == 5:

        #  ============================== PACKING MODEL v.2 ================================================

        num_classes = cbin_train.shape[1]
        indexes_premises_byclass = []
        for i in range(num_classes):  # debo de colocar aqui el numero de clases
            container_aux = []
            container_w = []
            for j in classifiers:
                container_aux.append(j[i][0])
                container_w.append(j[i][1].tolist())

            list_premises_container = list(chain(*container_aux))
            list_w_container = sum(list(chain(*container_w)), [])
            container_dic = dict(zip(list_premises_container, list_w_container))
            rule_duplicate = list(list_duplicates(list_premises_container))
            for rule in rule_duplicate:
                aver_w = sum(array(list_w_container)[rule[1]])/len(rule[1])
                container_dic[rule[0]] = aver_w
            unique_indexes = list(container_dic.keys())
            aux_unique_w = array(container_dic.values())/sum(container_dic.values())
            unique_w = aux_unique_w.reshape(aux_unique_w.shape[0], 1)
            indexes_premises_byclass.append([unique_indexes, unique_w, 'MQR'])

        output_classifier = eval_classifier_one(indexes_premises_byclass, ux_train, t_norm)  # Out U by class train
        output_bin = decision_class(output_classifier, freq_classes)

        final_premises_classes = []
        for i in range(len(indexes_premises_byclass)):  # x cada clase
            final_premises_classes.append(indexes_premises_byclass[i][0])

        F6 = Evaluation(indexes_premises_byclass, final_premises_classes, freq_classes)
        metrics_train = F6.eval_train(cbin_train, output_bin)
        metrics_test = F6.eval_test(cbin_test, ux_test, t_norm)
        metrics_rules = metrics_test[4]

        metrics = [1, [metrics_train[0] * 100, metrics_test[0] * 100, metrics_train[1], metrics_test[1],
                       metrics_rules],
                   successful_classifiers]
        results = [successful_classifiers, metrics_train[0] * 100, metrics_test[0] * 100, metrics_train[1],
                   metrics_test[1],
                   metrics_rules[0], metrics_rules[1]]

        report = toolfis.template_results('', results)
        return report, metrics, cv_i


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def estimation_average(u_estimation_class):
    u_mean = sum(u_estimation_class, 0)/len(u_estimation_class)
    return normalizar(u_mean)


def voting(out_bin):
    frequency = sum(out_bin, 0)
    return frequency


def estimation_weight(outputs_tree, cbin):
    container_ac = []
    for one_tree in outputs_tree:
        ac_classifier_i = eval_metrics(cbin, one_tree)
        container_ac.append(ac_classifier_i[0])
    return array(container_ac/sum(container_ac))


def rank_classifiers(acc, num_clf, classifiers_colector):
    if not acc in classifiers_colector:
        classifiers_colector[acc] = [num_clf]
    else:
        classifiers_colector[acc].append(num_clf)


def estimation_weight_by_class(outputs_tree, cbin):
    container_w_classes = []
    for one_tree in outputs_tree:  # By classifier
        classifier_i = eval_metrics(cbin, one_tree)
        matrix_confusion = classifier_i[2]
        sum_matrix_confusion = sum(matrix_confusion)
        ac_classes = []
        for i in range(len(sum_matrix_confusion)):  # by number classes
            if sum_matrix_confusion[i] == 0:
                ac_classes.append(float(matrix_confusion[i][i]) / 0.0001)
            else:
                ac_classes.append(float(matrix_confusion[i][i]) / (sum_matrix_confusion[i]))
        container_w_classes.append(array(ac_classes)/sum(ac_classes))
    return container_w_classes


def decision_class(output_tree, freq_classes):
    estimation_class = Decisions(output_tree, freq_classes)
    return estimation_class.dec_max_pert()


def aggregation_classifiers_by_weight(weight_by_classifier, outputs_tree):
    for i in range(len(weight_by_classifier)):
        outputs_tree[i] = outputs_tree[i]*weight_by_classifier[i]
    return sum(outputs_tree)


def aggregation_classifiers_by_weight_class(weight_class_by_classifier, outputs_tree):
    for i in range(len(weight_class_by_classifier)):
        outputs_tree[i] = outputs_tree[i]*weight_class_by_classifier[i]
    return sum(outputs_tree)


