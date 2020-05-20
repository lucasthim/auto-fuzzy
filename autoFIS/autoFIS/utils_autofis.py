from sklearn.utils import resample
from itertools import chain
import random

import numpy as np
import os
import math

from skfeature.function.statistical_based import CFS
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

from .lecture import Lecture
from .fuzzification import Fuzzification
from .decisions import Decisions

from .aggregation import Aggregation
from .association import Association
from .formul.formulation import Formulation
from .parameters_init import GlobalParameter

def dummies2int(binary_classes):
    return [i.index(1) + 1 for i in binary_classes.tolist()]


def random_distribution(tuple_item_weight):
    random_number = random.uniform(0, 1)
    s = 0
    for item, prob in tuple_item_weight:
        s += prob
        if s >= random_number:
            return item
    return tuple_item_weight[1][0]


def relative2absolute(list_indexes_premises, conversor):
    # list_indexes_premises: [(0,2), (3,), (7,9)]
    # conversor: {0:47, 1:48, 2:49}
    list_absolute_indexes_premises = []
    for relative_indexes_premise in list_indexes_premises:
        absolute_indexes_premise = tuple(
            conversor[j] for j in relative_indexes_premise)
        list_absolute_indexes_premises.append(absolute_indexes_premise)
    return list_absolute_indexes_premises


def gather_numbers_by_sizes(list_numbers, sizes_attributes):
    # list_numbers:     [0,1,2,3...,7]
    # sizes_attributes: [3, 2, 3]
    # output:           [(0,1,2), (3,4), (5,6,7)]
    new_list = []
    ref = 0
    for i in sizes_attributes:
        new_list.append(tuple(list_numbers[ref:ref+i]))
        ref += i
    return new_list


def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2


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
    return np.hstack(new_ux_prev)  # new_ux


def data_blb(ux_train_blb, y_bin_blb):
    p_r = GlobalParameter()
    size_ux_train_blb = np.shape(ux_train_blb)[0]
    integer_class = dummies2int(y_bin_blb)
    classes, size = list(np.unique(integer_class, return_counts=True))
    # index_by_class = np.array([])
    # for c in classes:
    #     index_clas_c = np.where(integer_class == c)
    #     size_class_c = index_clas_c[0].shape[0]
    #     size_resample = np.floor(pow(size_class_c, p_r.percent_resample)).astype(int)
    #     index_c = resample(index_clas_c[0], replace=False, n_samples=int(size_resample))
    #     index_oob = np.array(list(set(list(index_clas_c[0])) - set(index_c)))
    #     index_by_class = np.append(index_by_class, index_c).astype(int)
    # ux_blb = ux_train_blb[index_by_class, :]
    # new_y_bin_blb = y_bin_blb[index_by_class, :]
    class_num = 0
    size_resample_ux_train = np.floor(
        pow(size_ux_train_blb, p_r.percent_resample)).astype(int)
    index_ux_train_blb = range(size_ux_train_blb)
    while class_num == 0:
        index_resample = resample(
            index_ux_train_blb, replace=False, n_samples=size_resample_ux_train)
        resample_y_bin_blb = y_bin_blb[index_resample, :]
        integer_class = dummies2int(resample_y_bin_blb)
        classes_blb = list(np.unique(integer_class))
        if len(classes_blb) == len(classes):
            class_num = 1
            index_oob = np.array(
                list(set(list(index_ux_train_blb)) - set(list(index_resample))))
            ux_blb = ux_train_blb[index_resample, :]
            new_y_bin_blb = y_bin_blb[index_resample, :]
            # print ("Successful BLB")
    return [ux_blb, new_y_bin_blb, index_oob]


def data_blb_v2(ux_train_blb, y_bin_blb):
    p_r = GlobalParameter()
    size_ux_train_blb = np.shape(ux_train_blb)[0]
    integer_class = dummies2int(y_bin_blb)
    classes, size = list(np.unique(integer_class, return_counts=True))

    class_num = 0
    size_resample_ux_train = np.floor(
        pow(size_ux_train_blb, p_r.percent_resample)).astype(int)
    index_ux_train_blb = range(size_ux_train_blb)
    while class_num == 0:
        index_resample = resample(
            index_ux_train_blb, replace=False, n_samples=size_resample_ux_train)
        resample_y_bin_blb = y_bin_blb[index_resample, :]
        integer_class = dummies2int(resample_y_bin_blb)
        classes_blb = list(np.unique(integer_class))
        if len(classes_blb) == len(classes):
            class_num = 1
            index_oob = np.array(
                list(set(list(index_ux_train_blb)) - set(list(index_resample))))
            ux_blb = ux_train_blb[index_resample, :]
            new_y_bin_blb = y_bin_blb[index_resample, :]
            # print ("Successful BLB")
    return [ux_blb, new_y_bin_blb, index_oob]


def create_data(ref_attributes, size_attributes, premises_by_attribute, premises_contain_negation, ux_train, y_bin):

    # Generation  new data with resample
    p_r = GlobalParameter()
    size_ux_train = np.shape(ux_train)[0]
    index_ux_train = range(size_ux_train)
    size_resample_ux_train = np.floor(pow(size_ux_train, 1)).astype(int)
    ux, new_y_bin = resample(
        ux_train, y_bin, replace=True, n_samples=size_resample_ux_train)

    # Selection features
    # num_random_features = int(pow(len(ref_attributes), 1))  # optional multipliers: (0.5, 1, 2)
    num_random_features = int(math.log(len(ref_attributes), 2) + 1)
    random_features = resample(
        range(len(ref_attributes)), replace=False, n_samples=num_random_features)
    sub_sizes_attributes = [size_attributes[i] for i in random_features]
    sub_premises_by_attribute = [premises_by_attribute[i]
                                 for i in random_features]
    sub_premises_contain_negation = [
        premises_contain_negation[i] for i in premises_contain_negation]

    columns = list(chain(*sub_premises_by_attribute))
    new_ux = ux[:, columns]

    # the most important is "columns"
    genesis = [columns, random_features, sub_premises_by_attribute]

    # To eval quickly in autoFIS
    new_ref_attributes = range(num_random_features)
    aux = range(sum(sub_sizes_attributes))
    new_premises_by_attribute = gather_numbers_by_sizes(
        aux, sub_sizes_attributes)

    sub_data = [new_ux, new_y_bin, new_ref_attributes, new_premises_by_attribute,
                sub_sizes_attributes, sub_premises_contain_negation]

    return [sub_data, genesis]


def create_data_v2(ref_attributes, size_attributes, premises_by_attribute, premises_contain_negation, ux_train, y_bin):

    # Generation new data without resample
    # p_r = GlobalParameter()
    # size_ux_train = np.shape(ux_train)[0]
    # size_resample = int(pow(size_ux_train, p_r.percent_resample))
    #
    # ux, new_y_bin = resample(ux_train, y_bin, replace=True)
    # ux, new_y_bin = resample(ux_train, y_bin, replace=False, n_samples=size_resample)
    # ux, _, new_y_bin, _ = train_test_split(ux_train, y_bin, test_size=int(size_ux_train-size_resample),
    #                                        stratify=y_bin)

    # Selection features
    # clf = ExtraTreesClassifier()
    # clf = clf.fit(ux, dummies2int(new_y_bin))
    # scores = clf.feature_importances_
    #
    # soma = 0
    # num_features = 0
    # sortsc = np.sort(scores)
    # for i in reversed(range(scores.shape[0])):
    #     soma += sortsc[i]
    #     num_features += 1
    #     if soma >= 0.5:
    #         break
    #
    # features = list(scores.argsort()[-num_features:][::-1])
    #
    #
    # pre_random_features = []
    # for i in features:
    #     tupl = [item for item in premises_by_attribute if i in item]
    #     if len(tupl):
    #         pre_random_features.append(premises_by_attribute.index(tupl[0]))
    # random_features = list(set(pre_random_features))
    # num_random_features = len(random_features)

    num_random_features = int(math.log(len(ref_attributes), 2) + 1)
    random_features = resample(
        range(len(ref_attributes)), replace=False, n_samples=num_random_features)

    sub_sizes_attributes = [size_attributes[i] for i in random_features]
    sub_premises_by_attribute = [premises_by_attribute[i]
                                 for i in random_features]
    sub_premises_contain_negation = [
        premises_contain_negation[i] for i in premises_contain_negation]

    columns = list(chain(*sub_premises_by_attribute))
    # new_ux = ux[:, columns]
    new_ux = ux_train[:, columns]

    # the most important is "columns"
    genesis = [columns, random_features, sub_premises_by_attribute]

    # To eval quickly in autoFIS
    new_ref_attributes = range(num_random_features)
    aux = range(sum(sub_sizes_attributes))
    new_premises_by_attribute = gather_numbers_by_sizes(
        aux, sub_sizes_attributes)

    # sub_data = [new_ux, new_y_bin, new_ref_attributes, new_premises_by_attribute,
    #              sub_sizes_attributes, sub_premises_contain_negation]
    #
    sub_data = [new_ux, y_bin, new_ref_attributes, new_premises_by_attribute,
                sub_sizes_attributes, sub_premises_contain_negation]

    return [sub_data, genesis]


def divide_data(zipFilePath, file_train, file_test):

    reader = Lecture()
    reader.read_1cv(zipFilePath, file_train, file_test)
    x, y_bin, freq_classes, _, _, index_train = reader.info_data()

    matrix_x = x.copy()
    vec_y_bin = y_bin.copy()
    # Clustering
    list_scr = []
    rng = np.arange(2, 16, 2)

    matrix_x_norm = zscore(matrix_x, ddof=1)
    # matrix_x_norm = matrix_x.copy()
    for nc in range(rng.shape[0]):
        kmeans = KMeans(n_clusters=rng[nc]).fit(matrix_x_norm)
        labels = kmeans.labels_
        list_scr.append(silhouette_score(matrix_x_norm, labels))
    clt = list_scr.index(max(list_scr))
    print(str(rng[clt]) + ' clusters')

    kmeans_opt = KMeans(n_clusters=rng[clt]).fit(matrix_x_norm)
    labels_opt = kmeans_opt.labels_
    centers_opt = kmeans_opt.cluster_centers_
    counter = np.bincount(labels_opt)
    print(counter)
    #
    clt_peq = np.where(counter < int(0.01*matrix_x_norm.shape[0]))[0]
    centers_dist = squareform(pdist(centers_opt, metric='euclidean'))

    train_clt, test_clt = [], []
    for ic in range(rng[clt]):

        cluster = np.where(labels_opt == ic)[0]
        matrix_x_clu = matrix_x[cluster, :]
        vec_y_bin_x_clu = vec_y_bin[cluster, :]

        i_trn = np.where(cluster < index_train)[0]
        i_tst = np.where(cluster >= index_train)[0]

        train_clt.append([matrix_x_clu[i_trn, :], vec_y_bin_x_clu[i_trn, :]])
        test_clt.append([matrix_x_clu[i_tst, :], vec_y_bin_x_clu[i_tst, :]])

    # INCOMPLETO - falta implementar a fusao de grupos pequenos aos de centroides mais proximos
    #              Verificar entradas de todas as classes em cada cluster

    return train_clt, test_clt


def lecture_fuz_one_cv(zipFilePath, file_train, file_test, parameters):
    # Parameters
    cat_bool, fz_type, fz_number_partition = parameters[0:3]
    is_enable_negation = parameters[4]
    # Lecture
    reader = Lecture()
    reader.read_1cv(zipFilePath, file_train, file_test)
    # [x, y_Bin, Freq_Class, Dic_Labels, Dic_Class, Index_Train]
    x, y_bin, freq_classes, _, _, index_train = reader.info_data()

    # Fuzzification
    matrix_x = x.copy()
    fuzzifier = Fuzzification(matrix_x, cat_bool)
    fuzzifier.build_uX(fz_type, fz_number_partition)
    if is_enable_negation == 1:
        fuzzifier.add_negation()
    print('Successful FZ')
    # Getting train and test partitions
    ux_train = fuzzifier.uX[:index_train, :]
    ux_test = fuzzifier.uX[index_train:, :]
    cbin_train = y_bin[:index_train, :]
    cbin_test = y_bin[index_train:, :]

    # Information about attributes fuzzification
    sizes_attributes = fuzzifier.num_of_premises_by_attribute  # [3, 2, 3]
    # [(0,1,2),(3,4),(5,6,7)]
    premises_by_attribute = fuzzifier.attribute_premises
    ref_attributes = fuzzifier.ref_attributes  # [0, 1, 2]
    premises_contain_negation = fuzzifier.indexes_premises_contain_negation

    fuz_train = [ux_train, cbin_train]
    fuz_test = [ux_test, cbin_test]
    attributes_information = [
        sizes_attributes, premises_by_attribute, ref_attributes, premises_contain_negation]
# return fuz_train, fuz_test, attributes_information, freq_classes, gain_by_att
    return fuz_train, fuz_test, attributes_information, freq_classes


def lecture_fuz_one_cv_v2(train, test, parameters):
    # Parameters
    cat_bool, fz_type, fz_number_partition = parameters[0:3]
    is_enable_negation = parameters[4]

    x = np.vstack((train[0], test[0]))
    y_bin = np.vstack((train[1], test[1]))
    index_train = train[0].shape[0]

    # Fuzzification
    matrix_x = x.copy()
    fuzzifier = Fuzzification(matrix_x, cat_bool)
    fuzzifier.build_uX(fz_type, fz_number_partition)
    if is_enable_negation == 1:
        fuzzifier.add_negation()
    print('Successful FZ')
    # Getting train and test partitions
    ux_train = fuzzifier.uX[:index_train, :]
    ux_test = fuzzifier.uX[index_train:, :]
    cbin_train = y_bin[:index_train, :]
    cbin_test = y_bin[index_train:, :]

    # Information about attributes fuzzification
    sizes_attributes = fuzzifier.num_of_premises_by_attribute  # [3, 2, 3]
    # [(0,1,2),(3,4),(5,6,7)]
    premises_by_attribute = fuzzifier.attribute_premises
    ref_attributes = fuzzifier.ref_attributes  # [0, 1, 2]
    premises_contain_negation = fuzzifier.indexes_premises_contain_negation

    fuz_train = [ux_train, cbin_train]
    fuz_test = [ux_test, cbin_test]
    attributes_information = [
        sizes_attributes, premises_by_attribute, ref_attributes, premises_contain_negation]

    y = np.array(dummies2int(y_bin))
    freq = np.bincount(y)
    freq_classes = [1. * x / sum(freq) for x in freq[1:]]

    return fuz_train, fuz_test, attributes_information, freq_classes


# Formulation, Association, Aggregation, Decisions
def inference_fuzzy(data_fuz, parameters, info, ensemble="BagFIS"):
    # En la salida de samudio es los grados de pertinencia por clase
    # Para Jorge es binario
    ux, y_bin, ref_attributes, premises_by_attribute, num_premises_by_attribute, premises_contain_negation = data_fuz
    try:
        form = Formulation(ux, y_bin, ref_attributes, premises_by_attribute,
                           num_premises_by_attribute, premises_contain_negation)
        # Inputs given by user
        max_size_premises, t_norm, par_area, par_over, par_pcd = parameters[0:-3]
        association_method, aggregation_method = parameters[-3:-1]
        freq_classes = parameters[-1]

        arbol = form.gen_ARB(max_size_premises, t_norm,
                             par_area, par_over, par_pcd)

        status = [0 if not i[0] else 1 for i in arbol]
        sum_status = sum(status)
        if sum_status != len(arbol):
            if sum_status == 0:
                raise ValueError("Error in Formulation Module. Any premise survived. "
                                 "Sorry, you can not continue in the next stage."
                                 "\nTry to change the configuration")
            else:
                arb = [i for i in arbol if i[0]]
                arbol, arb = arb, arbol

        number_classes = y_bin.shape[1]

        # 4. Association
        f3 = Association(arbol, y_bin)
        premises_by_class = f3.division(association_method)

        # Verification classes without premises
        status = [0 if not i[0] else 1 for i in premises_by_class]
        if sum(status) != y_bin.shape[1]:
            raise ValueError("Error in Division Module. Some classes did not get premises. "
                             "Sorry, you can not continue in the next stage."
                             "\nTry to change the configuration")

        # 5. Aggregation:
        f4 = Aggregation(premises_by_class, y_bin, info)
        output_aggregation = f4.aggregation(aggregation_method)

        # 0: [[premises],weights,method_name]
        premises_weights_names = output_aggregation[0]
        estimation_classes = output_aggregation[1]  # 1: U_pertinence grade

        status = [0 if not i[0] else 1 for i in premises_weights_names]
        if sum(status) != number_classes:
            raise ValueError("Error in Aggregation Module. Some classes did not get premises. "
                             "Sorry, you can not continue in the next stage."
                             "\nTry to change the configuration")
        else:
            successful = 1
            f5 = Decisions(estimation_classes, freq_classes)
            train_bin_prediction = f5.dec_max_pert()

            outputs = [premises_weights_names, train_bin_prediction]
            if ensemble == "RandomFIS":
                outputs.append(estimation_classes)

    except ValueError as e:
        print(e)
        successful = 0
        outputs = ["The process was stopped in some stage"]

    return successful, outputs


def classifiers_aggregation(indexes_premises_by_class, y_bin, aggregation_method, freq_classes, info):
    number_classes = len(indexes_premises_by_class)
    try:
        f4 = Aggregation(indexes_premises_by_class, y_bin, info)
        output_aggregation = f4.aggregation(aggregation_method)

        # 0: premises, 1: weights
        premises_weights_names = output_aggregation[0]
        estimation_classes = output_aggregation[1]

        status = [0 if not i[0] else 1 for i in premises_weights_names]
        if sum(status) != number_classes:
            raise ValueError("Error in Aggregation Module. Some classes did not get premises. "
                             "Sorry, you can not continue in the next stage."
                             "\nTry to change the configuration")
        else:
            successful = 1
            f5 = Decisions(estimation_classes, freq_classes)
            train_bin_prediction = f5.dec_max_pert()
            outputs = [output_aggregation[0],
                       train_bin_prediction, estimation_classes]
    except ValueError as e:
        print(e)
        # report = e  # .append("\n" + str(e))
        successful = 0
        outputs = ["No se termino el proceso, se detuvo en algun etapa"]
    return successful, outputs


def create_error_file(root, zip_file_name, message, i_cv):
    name_error = os.path.join(root, 'ERROR' + zip_file_name[:-13])
    fail_error = open(name_error, 'w')
    fail_error.write('Error in CV:' + str(i_cv + 1))
    fail_error.write("\n" + message[0])
    fail_error.close()


def create_csv_summaries(root):
    csv_summary = os.path.join(root, "Resumo_Ac_Times.csv")
    file_res = open(csv_summary, 'w')
    file_res.write(
        "Dataset, Accuracy, Std(Ac), Num_Regras, Successful_Classifiers, Time(s)\n")
    file_res.close()
    return csv_summary


def write_summaries(csv_ac_nr_nums, data_name_key, out_results):
    file_res = open(csv_ac_nr_nums, 'a')
    row_data = (data_name_key,) + out_results
    file_res.write('%s, %.2f, %.2f, %.2f, %.2f, %.2f\n' % row_data)
    file_res.close()


def get_formulation_parameters(parameters):
    criteria_area = parameters[6]
    area_threshold = parameters[7]
    is_enable_pcd = parameters[8]
    is_enable_overlapping = parameters[9]
    overlapping_threshold = parameters[10]
    par_area = [criteria_area, area_threshold]
    par_over = [is_enable_overlapping, overlapping_threshold]
    par_pcd = is_enable_pcd
    return par_area, par_over, par_pcd


def summary_multiple(successful_classifiers, metrics_train, metrics_test, metrics_rules):
    r0 = successful_classifiers
    ac_train, auc_train = metrics_train[0:2]
    ac_test, auc_test = metrics_test[0:2]
    r1 = "%.4f, %.4f" % (np.mean(ac_train), np.std(ac_train))
    r2 = "%.4f, %.4f" % (np.mean(ac_test), np.std(ac_test))
    r3 = "%.4f, %.4f" % (np.mean(auc_train), np.std(auc_train))
    r4 = "%.4f, %.4f" % (np.mean(auc_test), np.std(ac_test))
    r5 = np.mean(metrics_rules[0])
    r6 = np.mean(metrics_rules[1])
    r7 = r6 / r5
    return r0, r1, r2, r3, r4, r5, r6, r7


def template_results(nome, results):  # Sirve para OneCV y OneZip
    r0, r1, r2, r3, r4, r5, r6, r7 = results
    summary = """
    {x}
    Successful classifiers: {x0}
    Accuracy training: {x1}
    Accuracy Testing: {x2}
    AUC Training: {x3}
    AUC Testing: {x4}
    Number of Rules: {x5}
    Total Rule Length: {x6} / Average: {x7:.4f}
    """.format(x=nome, x0=r0, x1=r1, x2=r2, x3=r3, x4=r4, x5=r5, x6=r6, x7=float(r7))
    return summary
