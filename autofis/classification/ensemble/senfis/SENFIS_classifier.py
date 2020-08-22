import os
from autoFIS.autoFIS.databases_map import dictionary_data
import autoFIS.autoFIS.utils_autofis as toolfis
from autoFIS.autoFIS.evaluation import Evaluation
from itertools import chain
import timeit
import datetime
from RandomFISonezip import random_fis_one_zip
from evalRF import normalizar
from parameters_init import GlobalParameter


def define_parameters(database_name, pars):
    try:
        parameters = [dictionary_data(database_name)[0]] + pars  # addition of 2 list
    except KeyError:
        print ("The database " + database_name + " was not found.\nIt was assumed that all attributes are numeric")
        parameters = [0]
    return parameters


def pack_model_cvs(packaging_parameters):

    successful_classifiers = packaging_parameters[0]
    classifiers, cbin_train, ux_train, cbin_test, ux_test, t_norm, freq_classes = packaging_parameters[1]

    num_classes = cbin_train.shape[1]
    indexes_premises_byclass = []
    unique_indexes_byclass = []
    for i in range(num_classes):
        container_aux = []
        for j in classifiers:
            container_aux.append(j[i][0])

        list_premises_container = list(chain(*container_aux))
        unique_indexes = list(set(list_premises_container))
        unique_indexes_byclass.append(unique_indexes)
        unique_premises = toolfis.calculation_premises(unique_indexes, ux_train, t_norm)
        indexes_premises_byclass.append([unique_indexes, unique_premises])

    exito, output_collective = toolfis.classifiers_aggregation(indexes_premises_byclass, cbin_train, 'MQR', freq_classes)

    if not exito:
        report, metrics = ["Final aggregation was fail"], [0]
    else:
        premises_weights_names, estimation_classes = output_collective

        final_premises_classes = []
        for i in range(len(premises_weights_names)):  # x cada clase
            final_premises_classes.append(premises_weights_names[i][0])

        f6 = Evaluation(premises_weights_names, final_premises_classes, freq_classes)
        metrics_train = f6.eval_train(cbin_train, estimation_classes)
        metrics_test = f6.eval_test(cbin_test, ux_test, t_norm)

        results_cv = summary_cv_bagfis(successful_classifiers, metrics_train, metrics_test)
        report = toolfis.template_results(results_cv)

        metrics = [1, [metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1], metrics_test[4]]]
    return report, metrics


def estimation_u_by_class(u_estimation_class):
    u_mean = sum(u_estimation_class, 0)/len(u_estimation_class)
    return normalizar(u_mean)


def summary_cv_bagfis(successful_classifiers, metrics_train, metrics_test):
    r0 = successful_classifiers
    r1, r3 = metrics_train[0:2]
    r2, r4 = metrics_test[0:2]
    r5 = metrics_test[4][0]
    r6 = metrics_test[4][1]
    return r0, r1, r2, r3, r4, r5, r6


def packing_train_test_files(files_cv):
    if len(files_cv) == 20:  # number_files_zip
        a = files_cv[2:] + files_cv[0:2]
    else:  # number_files_zip == 10
        a = files_cv
    return a[::2], a[1::2]  # list_train_csv, list_test_csv


def container_results(number_cv_pairs):
    ac_train = number_cv_pairs * [0]
    ac_test = number_cv_pairs * [0]
    auc_train = number_cv_pairs * [0]
    auc_test = number_cv_pairs * [0]
    num_rules = number_cv_pairs * [0]
    total_rule_length = number_cv_pairs * [0]
    successful_classifiers = number_cv_pairs * [0]
    return ac_train, ac_test, auc_train, auc_test, num_rules, total_rule_length, successful_classifiers

# TODO: Método principal!!!
def run_random_fis(root, data, autofis_parameters, report_folder, clf_n, folder__tree_outpus):
    t0 = timeit.default_timer()
    try:
        data_name_key = data[0:-16]
        parameters_database = define_parameters(data_name_key, autofis_parameters)
        print('')
        print('parameters_database:',parameters_database)
        print('clf_n:',clf_n)
        print('data_name_key:',data_name_key)
        print('------------------------------------')
        achievement, out_results = random_fis_one_zip(root, data, parameters_database, report_folder, clf_n,
                                                      folder__tree_outpus, data_name_key, t0)
        if achievement == 0:
            raise ValueError("Problems in database: " + "<" + data + ">")
    except ValueError as e:
        achievement = 0
        out_results = ""
        print('Problem running RandomFIS.')
        print (e)
    tf = timeit.default_timer()
    time_end = tf - t0
    out_results = list(out_results)
    out_results.append(time_end)
    return achievement, out_results


def eval_architectures(list_architectures, folder_report, data_folder, datasets, clf_n):
    csv_summary = toolfis.create_csv_summaries(folder_report)
    folder__tree_output = os.path.join(folder_report, 'tree_outputs')
    os.makedirs(folder__tree_output)
    for data in datasets:
        is_successful, results_and_time = run_random_fis(data_folder, data, list_architectures, folder_report, clf_n,
                                                         folder__tree_output)
        if is_successful:
            data_name_key = data[0:-16]
            results = tuple(results_and_time)
            toolfis.write_summaries(csv_summary, data_name_key, results)


def eval_n_classifiers(folder_databases, databases, dic_architectures, n_classifiers=[75]):
    folder_datetime = os.path.join(os.getcwd(),'reports', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(folder_datetime)
    for i in n_classifiers:  # i = 75
        folder_i_classifiers_evaluated = os.path.join(folder_datetime, str(i))
        os.makedirs(folder_i_classifiers_evaluated)
        list_architectures = dic_architectures[i]
        print('n_classifiers', n_classifiers)
        eval_architectures(list_architectures, folder_i_classifiers_evaluated, folder_databases, databases, i)


def load_parameters_random_classifiers(n_classf, bin, percent_oob, percet_resample, enable_oob):

    # Fuzzification Parameters |
    triangular_fuzzy_type = "tukey"  # "normal"
    num_partitions_by_attribute = 3 # 3, 5
    t_norm = "prod"  # "min"

    # -----------------------|
    # Formulation parameters |

    # Negation whit size_premises
    enable_negation = 1
    size_premises = 2

    # Area filter parameters:
    criteria_area = "frequencia_relativa"  #  "cardinalidade_relativa",
    area_threshold = 0.075

    # PCD filter parameter:
    if bin == 1:
        is_enable_pcd = [1, 1]
    else:
        is_enable_pcd = [0, 0]

    # Overlapping filter parameters:
    is_enable_overlapping = [1, 1]
    overlapping_threshold = 0.95

    # ------------------------|
    # Association - Splitting |
    method_splitting = "MQR"

    # -------------|
    #  Aggregation |
    method_aggregation = 'MQR'

    parameters = [triangular_fuzzy_type,
                  num_partitions_by_attribute,
                  t_norm,
                  enable_negation,
                  size_premises,
                  criteria_area,
                  area_threshold,
                  is_enable_pcd,
                  is_enable_overlapping,
                  overlapping_threshold,
                  method_splitting,
                  method_aggregation,
                  n_classf,
                  percent_oob,
                  percet_resample,
                  enable_oob]
    return [parameters]


def get_list_databases(folder_databases):
    files = os.listdir(folder_databases)
    return [i for i in files if i.endswith("_csv.zip")]


def main():
    # RandomFIS parameters
    param = GlobalParameter()
    type_data = param.t_data
    n_classifiers_parameters = {}
    for i in param.n_classifiers:
        n_classifiers_parameters[i] = load_parameters_random_classifiers(i, type_data, param.method_aggregation,
                                                                         param.percent_resample,  param.blb)
    root_databases = os.path.dirname(os.path.realpath(__file__))
    databases = get_list_databases(root_databases)
    # print('Clf parameters:')
    # print(n_classifiers_parameters)

    eval_n_classifiers(root_databases, databases, n_classifiers_parameters, param.n_classifiers)


if __name__ == '__main__':
    main()
