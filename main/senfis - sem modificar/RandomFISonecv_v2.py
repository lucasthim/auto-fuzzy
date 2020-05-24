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

from WAD_calc import wad_calc, wad_calc_v2, reg_calc
from operator import itemgetter
from sklearn.model_selection import StratifiedKFold
import numpy as np

import _thread
import _threading_local


def random_fis_one_cv(zipFilePath, file_train, file_test, parameters_classifiers, cv_i, clf_n):

    # General parameters
    cv_i += 1
    print()
    print('Clf = %i, CV = %i' % (clf_n, cv_i))
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


    param = GlobalParameter()
    for i in range(parameters[-4]):

        ux_train_blb, new_y_bin_blb, index_oob = toolfis.data_blb(ux_train, cbin_train)

        classifiers_blb = []

        n_blb = list(range(param.blb))

        for blb_i in n_blb:
            new_data_blb, genesis_data_blb = toolfis.create_data(ref_attributes, sizes_attributes,
                                                                 premises_by_attribute,
                                                                 premises_contain_negation,
                                                                 ux_train_blb, new_y_bin_blb)
            exit_flag_blb, out_model_blb = toolfis.inference_fuzzy(new_data_blb, pars, info=(str(blb_i), str(i+1)),
                                                                   ensemble='RandomFIS')
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
                                                             freq_classes, info=('All models', str(i+1)))

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

    if classifiers:
        return ux_train, ux_test, cbin_train, cbin_test, freq_classes, parameters_classifiers, classifiers, cv_i,\
               outputs_tree_train, outputs_tree_train_bin, container_ac_train,\
               outputs_tree_test, outputs_tree_test_bin, partial_metrics_rules
    elif not classifiers:
        return ["Any of %i classifiers was successful" % successful_classifiers], [0], cv_i


#  ============================== CLASSIFIERS SELECTION ==================================================

def selection_criteria(ux_train, ux_test, cbin_train, cbin_test, freq_classes, parameters_classifiers, classifiers,
                       cv_i, outputs_tree_train, outputs_tree_train_bin, container_ac_train,
                       outputs_tree_test, outputs_tree_test_bin, partial_metrics_rules,
                       sel_method, sel_param, size_ensemble, folder__tree_output, time_ini):

    param_clf = [parameters_classifiers[0]] + parameters_classifiers[1]
    t_norm = param_clf[3]

    # Objective function
    if sel_method == 0:
        sel_name = 'WAD'
        wad_values = wad_calc_v2(cbin_train, outputs_tree_train_bin, container_ac_train, [sel_param, 1])
        wad_list = enumerate(wad_values)
        clf_sorted = sorted(wad_list, key=itemgetter(1))
    else:
        sel_name = 'REG'
        reg_values = reg_calc(cbin_train, outputs_tree_train_bin, container_ac_train, [sel_param, 1])
        reg_list = enumerate(reg_values)
        clf_sorted = sorted(reg_list, key=itemgetter(1))

    # Selection of N classifiers
    clf_sel_list = [int(i[0]) for i in clf_sorted[-size_ensemble:]]
    sel_classifiers = [classifiers[i] for i in clf_sel_list]

    # Selected classifiers results
    sel_outputs_tree_train = [outputs_tree_train[i] for i in clf_sel_list]
    sel_outputs_tree_train_bin = [outputs_tree_train_bin[i] for i in clf_sel_list]

    sel_outputs_tree_test = [outputs_tree_test[i] for i in clf_sel_list]
    sel_outputs_tree_test_bin = [outputs_tree_test_bin[i] for i in clf_sel_list]

    sel_partial_metrics_rules = [partial_metrics_rules[i] for i in clf_sel_list]
    sel_successful_classifiers = [len(clf_sel_list), sel_name, sel_param]

    p_method_agg = GlobalParameter()
    method_aggreg_classf = p_method_agg.method_aggregation

    if method_aggreg_classf == 4:

        #  ============================== PACKING ONE MODEL V.1 ============================================

        #  Evaluation Test_average
        u_by_class_ave = estimation_average(sel_outputs_tree_test)
        test_decision_ave = decision_class(u_by_class_ave, freq_classes)
        metrics_test_ave = round(eval_metrics(cbin_test, test_decision_ave)[0], 2)

        voting_bin = voting(sel_outputs_tree_test_bin)
        vot_bin = decision_class(voting_bin, freq_classes)
        metrics_test_vot = round(eval_metrics(cbin_test, vot_bin)[0], 2)

        #  Evaluation rules
        aux_metrics_rules_ave = sum(sel_partial_metrics_rules, 0)
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
            for j in sel_classifiers:
                container_aux.append(j[i][0])

            list_premises_container = list(chain(*container_aux))
            unique_indexes = list(set(list_premises_container))
            unique_premises = toolfis.calculation_premises(unique_indexes, ux_train, t_norm)
            indexes_premises_byclass.append([unique_indexes, unique_premises])

        exito, output_collective = toolfis.classifiers_aggregation(indexes_premises_byclass, cbin_train, 'MQR'
                                                                   , freq_classes,
                                                                   info=('', str(sel_successful_classifiers)))

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
                       sel_successful_classifiers]
            results = [sel_successful_classifiers, metrics_train[0]*100, metrics_test[0]*100, metrics_train[1], metrics_test[1],
                       metrics_rules[0], metrics_rules[1], metrics_rules[2]]

            report = toolfis.template_results('', results)
            return report, metrics, cv_i, sel_successful_classifiers


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


