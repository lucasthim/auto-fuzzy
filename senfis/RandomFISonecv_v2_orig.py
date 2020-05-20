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

from sklearn.neighbors import NearestNeighbors
from numpy import array_equal
from scipy.spatial.distance import jaccard, minkowski
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold

from WAD_calc import wad_calc
from operator import itemgetter

import copy
import numpy as np

import _thread
import _threading_local


def random_fis_one_cv(zipFilePath, file_train, file_test, parameters_classifiers, cv_i, clf_n, folder__tree_output,
                      time_ini):

    # General parameters
    p_method_agg = GlobalParameter()
    method_aggreg_classf = p_method_agg.method_aggregation
    cv_i += 1
    # print ('Clf = %i, CV = %i' % (clf_n, cv_i))
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

    ux_train_clean = ux_train
    cbin_train_clean = cbin_train

    # attr_position = []
    # size_attr = np.asarray(sizes_attributes)
    # for i in range(size_attr.shape[0]):
    #     if i == 0:
    #         attr_position.append([i, size_attr[i], parameters_classifiers[0][i]])
    #     else:
    #         attr_position.append([sum(size_attr[0:i]), sum(size_attr[0:i]) + size_attr[i],
    #                               parameters_classifiers[0][i]])

    # sort_attr = []
    # attr_pos = np.asarray(parameters_classifiers[0])
    # i_real = np.where(attr_pos == 0)[0]
    # i_bool = np.where(attr_pos != 0)[0]
    # for i in i_real:
    #     sort_attr.append(premises_by_attribute[i])
    # for i in i_bool:
    #     sort_attr.append(premises_by_attribute[i])
    # attr_pre = list(sort_attr)
    # attr = [i for sub in attr_pre for i in sub]
    # if i_bool.shape[0] == 1:
    #     ind_bool = attr.index(np.asarray(premises_by_attribute[i_bool[0]]))
    # else:
    #     attr_bool = [i for sub in premises_by_attribute[i_bool[0]:(i_bool[-1]+1)] for i in sub]
    #     ind_bool = list(range(attr.index(attr_bool[0]), attr.index(attr_bool[-1])+1))
    #
    #
    # def minkowjac(x, y, **kwargs):
    #
    #     if kwargs["ind_bool"][1] > 1:
    #         d_minkow = minkowski(x[0:kwargs["ind_bool"][0][0]], y[0:kwargs["ind_bool"][0][0]], p=2)
    #     else:
    #         d_minkow = minkowski(x[0:kwargs["ind_bool"][0]], y[0:kwargs["ind_bool"][0]], p=2)
    #     d_jac = jaccard([x[kwargs["ind_bool"][0]] >= 0.5], y[kwargs["ind_bool"][0]])
    #     return d_minkow + d_jac


    #Identifying and purging similar entries - Safe / Danger
    # n_neigh = 8
    # # vec_bool = np.asarray(ind_bool)
    # # nbrs = NearestNeighbors(n_neighbors=n_neigh, algorithm='ball_tree',
    # #                         metric=minkowjac, metric_params={"ind_bool": [ind_bool,
    # #                                                          vec_bool.size]}, n_jobs=-1).fit(ux_train[:, attr])
    #
    # nbrs = NearestNeighbors(n_neighbors=n_neigh, algorithm='ball_tree',
    #                         metric='minkowski', p=2, n_jobs=-1).fit(ux_train)
    #
    # _, indices = nbrs.kneighbors(ux_train)
    # i_sf, i_dng = [], []
    # for ind in range(ux_train.shape[0]):
    #     c_sf, c_dng = 0, 0
    #     for indn in range(1, n_neigh):
    #         if array_equal(cbin_train[ind][-1], cbin_train[indices[ind, indn]][-1]) is True:
    #             c_sf += 1
    #         else:
    #             c_dng += 1
    #     if c_sf > c_dng:
    #         i_sf.append(ind)
    #     else:
    #         i_dng.append(ind)
    #
    # ux_train_clean = np.delete(ux_train, i_sf, axis=0)
    # cbin_train_clean = np.delete(cbin_train, i_sf, axis=0)


    # Identifying and purging similar entries - Neighborhood
    # nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(ux_train)
    # distances, indices = nbrs.kneighbors(ux_train)
    # d_thres = 0.05 * max(distances[:, 1])
    # i_thres = list(np.where(distances[:, 1] <= d_thres))
    # ind_unique = np.unique(indices[i_thres, 1])
    #
    # ux_train_clean = np.delete(ux_train, ind_unique, axis=0)
    # cbin_train_clean = np.delete(cbin_train, ind_unique, axis=0)


    # Develop diverse sets - Clustering
    # list_scr = []
    #
    # rng = np.arange(5, 31, 5)
    # for nc in range(rng.shape[0]):
    #     # kmeans = MiniBatchKMeans(n_clusters=rng[nc], batch_size=int(0.05*ux_train_clean.shape[0])).fit(ux_train_clean)
    #     kmeans = KMeans(n_clusters=rng[nc]).fit(ux_train_clean)
    #     labels = kmeans.labels_
    #     # list_scr.append(silhouette_score(ux_train_clean, labels, sample_size=int(0.05*ux_train_clean.shape[0])))
    #     list_scr.append(silhouette_score(ux_train_clean, labels))
    # clt = list_scr.index(max(list_scr))
    # print(str(rng[clt]) + ' clusters')
    #
    # histcs = []
    # # kmeans_a = MiniBatchKMeans(n_clusters=rng[clt], batch_size=int(0.05*ux_train_clean.shape[0])).fit(ux_train_clean)
    # kmeans_a = KMeans(n_clusters=rng[clt]).fit(ux_train_clean)
    # histcs.append(np.bincount(kmeans_a.labels_) / sum(np.bincount(kmeans_a.labels_)))
    #
    # # Clustering n BLB data
    # n_rep = 500
    # ux_train_blb, new_y_bin_blb, index_oob, histc_blb = [], [], [], []
    # for i_rep in range(n_rep):
    #     ux_train_new, new_y_bin, index_oob_new = toolfis.data_blb(ux_train_clean, cbin_train_clean)
    #     blbc = kmeans_a.fit_predict(ux_train_new)
    #     histcs.append(np.bincount(blbc) / sum(np.bincount(blbc)))
    #     ux_train_blb.append(ux_train_new)
    #     new_y_bin_blb.append(new_y_bin)
    #     index_oob.append(index_oob_new)
    # print("Successful " + str(n_rep) + " BLB")
    #
    # # Clustering similarity vectors
    # kmeans_b = KMeans(n_clusters=int(parameters[-4] + 1)).fit(histcs)
    # c_orig = kmeans_b.labels_[0]
    # c_array = kmeans_b.labels_
    #
    # # Identifying diverse BLB sets
    # i_sel_blb = []
    # for ic in range(parameters[-4] + 1):
    #     if ic == c_orig:
    #         continue
    #     ic_blb = list(np.where(c_array == ic)[0])
    #     d_ic_blb = kmeans_b.transform([histcs[i] for i in ic_blb])
    #     d_orig_blb = d_ic_blb[:, c_orig]
    #     i_blb = np.where(d_orig_blb == max(d_orig_blb))
    #     i_sel_blb.append(ic_blb[i_blb[0][0]])


    # ===== Classifiers =====

    # n_split = int(ux_train_clean.shape[0]/7500)
    # n_split = 5
    # print('Clf = %i, CV = %i' % (n_split, cv_i))
    #
    # freq = np.unique(toolfis.dummies2int(cbin_train), return_counts=True)
    # min_freq = min(freq[1])
    #
    # if min_freq < n_split:
    #     n_split = min_freq
    #
    # skf = StratifiedKFold(n_splits=n_split, shuffle=True)

    # kmeans = MiniBatchKMeans(n_clusters=n_split, batch_size=int(0.05*ux_train_clean.shape[0])).fit(ux_train_clean)
    # kmeans = KMeans(n_clusters=n_split).fit(ux_train_clean)
    # labels = kmeans.labels_

    i_fold = 0
    for i in range(parameters[-4]):
    # for trn_index, tst_index in skf.split(ux_train_clean, toolfis.dummies2int(cbin_train_clean)):
        i_fold += 1

        classifiers_blb = []

        # sel_ux_train_blb = ux_train_blb[i_sel_blb[i]]
        # sel_new_y_bin_blb = new_y_bin_blb[i_sel_blb[i]]

        # Clustering
        # i_label = list(np.where(labels == i)[0])
        # sel_ux_train_blb = ux_train_clean[i_label]
        # sel_new_y_bin_blb = cbin_train_clean[i_label]

        # sel_ux_train_blb, sel_new_y_bin_blb, index_oob = toolfis.data_blb_v2(ux_train_clean, cbin_train_clean)

        # sel_ux_train_blb = ux_train_clean[tst_index]
        # sel_new_y_bin_blb = cbin_train_clean[tst_index]

        sel_ux_train_blb = ux_train_clean
        sel_new_y_bin_blb = cbin_train_clean

        i_blb = 0
        n_blb = list(range(p_blb.blb))
        eval_classifiers_blb = []
        for blb_i in n_blb:
            i_blb += 1
            # new_data_blb, genesis_data_blb = toolfis.create_data(ref_attributes, sizes_attributes,
            #                                                      premises_by_attribute,
            #                                                      premises_contain_negation,
            #                                                      sel_ux_train_blb, sel_new_y_bin_blb)


            new_data_blb, genesis_data_blb = toolfis.create_data_v2(ref_attributes, sizes_attributes,
                                                                 premises_by_attribute,
                                                                 premises_contain_negation,
                                                                 sel_ux_train_blb, sel_new_y_bin_blb)


            exit_flag_blb, out_model_blb = toolfis.inference_fuzzy\
                (new_data_blb, pars, info=(str(i_blb), str(i_fold)), ensemble='RandomFIS')
            #  out_model = [premises_weights_names, train_bin_prediction, estimation_classes(u_estimation)]
            eval_classifiers_blb.append([out_model_blb[1], eval_metrics(sel_new_y_bin_blb, out_model_blb[1])[0]])

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

        mat_eval = np.asarray(eval_classifiers_blb)
        wad_values = wad_calc(sel_new_y_bin_blb, mat_eval[:, 0], mat_eval[:, 1], [0.5, 0.5, 1])
        wad_list = enumerate(wad_values)
        wad_sorted = sorted(wad_list, key=itemgetter(1))
        sel_list = [int(i[0]) for i in wad_sorted[-10:]] #10 melhores
        classifiers_blb_redux = list(classifiers_blb[i] for i in sel_list)


        num_classes = cbin_train.shape[1]
        indexes_premises_byclass = []
        for ci in range(num_classes):  # debo de colocar aqui el numero de clases
            container_aux = []
            # for j in classifiers_blb:
            for j in classifiers_blb_redux:
                container_aux.append(j[ci][0])

            list_premises_container = list(chain(*container_aux))
            unique_indexes = list(set(list_premises_container))
            unique_premises = toolfis.calculation_premises(unique_indexes, sel_ux_train_blb, t_norm)
            indexes_premises_byclass.append([unique_indexes, unique_premises])

        success, out_model = toolfis.classifiers_aggregation(indexes_premises_byclass, sel_new_y_bin_blb, 'MQR',
                                                             freq_classes, info=('All models', str(i_fold)))

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
            container_ac_train.append(eval_metrics(sel_new_y_bin_blb, out_model[1])[0])
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

        # vec_dng = np.array(i_dng)
        # redu_trn = vec_dng.shape[0] / ux_train.shape[0]

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

        # folder__tree_output_test = os.path.join(folder__tree_output, 'reducao_trn.txt')
        # with open(folder__tree_output_test, 'a') as f:
        #     f.write(str(redu_trn) + '\n')

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
                                                                   , freq_classes, info=('batches', 'All'))

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
                       metrics_rules[0], metrics_rules[1], metrics_rules[2]]

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


