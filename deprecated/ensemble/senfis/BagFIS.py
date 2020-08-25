__author__ = 'jparedes'

import os
from autoFIS.autoFIS.databases_map import dictionary_data
import autoFIS.autoFIS.utils_autofis as toolfis
from autoFIS.autoFIS.evaluation import Evaluation
from numpy import mean, std
from itertools import chain
import zipfile
import timeit
import datetime


def define_parameters(database_name, pars):
    try:
        parameters = [dictionary_data(database_name)[0]] + pars  # addition of 2 list
    except KeyError:
        print "The database " + database_name + " was not found.\nIt was assumed that all attributes are numeric"
        parameters = [0]
    return parameters


def BagFISonecv(zipFilePath, file_train, file_test, parameters, num_classifiers):
    # General parameters
    t_norm = parameters[3]
    max_size_of_premise = parameters[5]
    association_method = parameters[11]
    aggregation_method = parameters[12]

    # Gathering parameters
    # Formulation parameters:
    par_area, par_over, par_pcd = toolfis.get_formulation_parameters(parameters)

    # 1. Lecture & Fuzzification
    out1 = toolfis.lecture_fuz_one_cv(zipFilePath, file_train, file_test, parameters)
    ux_train, cbin_train = out1[0]
    ux_test, cbin_test = out1[1]
    sizes_attributes, premises_by_attribute, ref_attributes, premises_contain_negation = out1[2]
    freq_classes = out1[3]

    outputs_trees = []
    pars = [max_size_of_premise, t_norm, par_area, par_over, par_pcd,
            association_method, aggregation_method, freq_classes]
    classifiers = []
    successful_classifiers = 0

    # ================================= GENERATION CLASSIFIERS =========================================================

    for i in range(num_classifiers):
        new_data, genesis_data = toolfis.create_data(ref_attributes, sizes_attributes, premises_by_attribute,
                                                     premises_contain_negation, ux_train, cbin_train)
        exit_flag, salida = toolfis.inference_fuzzy(new_data, pars)
        # salida: [[premisesFinales, weights, method], [Train_prediction]]

        if exit_flag:
            successful_classifiers += 1
            outputs_trees.append(salida[1])  # Output Train

            # Transformation premises relative2absolute:
            converter = dict(zip(range(len(genesis_data[0])), genesis_data[0]))
            absolute_models = []
            for j in salida[0]:
                relative_premises = j[0]
                absolute_premises = toolfis.relative2absolute(relative_premises, converter)
                absolute_models.append([absolute_premises, j[1], j[2]])  # premises absolutes, Weights, name_method
            classifiers.append(absolute_models)

    # ======================================= PACKING IN ONE MODEL =================================================

    if not classifiers:
        return ["Any of %i classifiers was successful" % num_classifiers], [0]

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

    exito, output_collective = toolfis.classifiers_aggregation(indexes_premises_byclass, cbin_train, 'MQR',
                                                               freq_classes)

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

        metrics = [1, [metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1], metrics_test[4]],
                   successful_classifiers]

    return report, metrics


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


def BagFISonezip(root, zip_file_name, parameters, num_classifiers, report_folder):
    zipFilePath = os.path.join(root, zip_file_name)

    try:
        with zipfile.ZipFile(zipFilePath, 'r') as z:
            files_cv = z.namelist()

        number_files_zip = len(files_cv)
        if not (number_files_zip == 20 or number_files_zip == 10 or number_files_zip == 2):
            raise ValueError("This module works with a zip file to 10cv or 5cv. "
                             "For this reason, it is expected 20 or 10 files inside the zip file")
        number_cv_pairs = number_files_zip / 2
        list_train, list_test = packing_train_test_files(files_cv)

        msg = []
        ac_train, ac_test, auc_train, auc_test, num_rules, rule_len, suc_classifiers = container_results(
            number_cv_pairs)

        for i in range(number_cv_pairs):
            train_file = list_train[i]
            test_file = list_test[i]

            message, indicators = BagFISonecv(zipFilePath, train_file, test_file, parameters, num_classifiers)
            msg.append(message)

            if indicators[0] == 0:
                toolfis.create_error_file(root, zip_file_name, message, i)
                raise ValueError("Problem detected in CV " + str(i + 1))

            ac_train[i], ac_test[i] = indicators[1][0], indicators[1][1]
            auc_train[i], auc_test[i] = indicators[1][2], indicators[1][3]
            num_rules[i] = indicators[1][4][0]
            rule_len[i] = indicators[1][4][1]
            suc_classifiers[i] = indicators[2]

        filename_report = os.path.join(report_folder, 'Report of ' + zip_file_name[:-8])
        target = open(filename_report, 'w')
        summary_parameters = 'Parameters: %s' % str(parameters)
        underline = "\n%s\n" % (len(summary_parameters) * "=")

        cvs_info = [summary_parameters]

        for i in range(number_cv_pairs):
            cv_summary = """    CV-{indice}:
            {cvv}""".format(indice=i + 1, cvv=msg[i])
            cvs_info.append(cv_summary)

        results = toolfis.summary_multiple(mean(suc_classifiers), [ac_train, auc_train],
                                           [ac_test, auc_test], [num_rules, rule_len])
        abstract = toolfis.template_results(results)
        cvs_info.append(abstract)
        target.write(underline.join(cvs_info))
        target.close()

        achievement = 1
        print "win ", zip_file_name

    except ValueError as e:
        print e
        achievement, ac_test, num_rules, suc_classifiers = 0, 0, 0, 0

    return achievement, (100 * mean(ac_test), 100 * std(ac_test), mean(num_rules), mean(suc_classifiers))


def runBagFIS(root, data, autofis_parameters, number_classifiers, report_folder):
    t0 = timeit.default_timer()
    try:
        data_name_key = data[0:-16]
        parameters_database = define_parameters(data_name_key, autofis_parameters)
        achievement, out_results = BagFISonezip(root, data, parameters_database, number_classifiers, report_folder)
        if achievement == 0:
            raise ValueError("Problems in database: " + "<" + data + ">")
    except ValueError as e:
        achievement = 0
        out_results = ""
        print e
    tf = timeit.default_timer()
    return achievement, (out_results, tf - t0)


def eval_architectures(list_architectures, folder_report, data_folder, datasets, n_classifiers):
    for architecture in list_architectures:  # each architecture is an autofis parameters
        csv_times, csv_summary = toolfis.create_csv_summaries(folder_report)
        for data in datasets:
            is_successful, results_and_time = runBagFIS(data_folder, data, architecture, n_classifiers, folder_report)
            if is_successful:
                data_name_key = data[0:-16]
                results, time_elapsed = results_and_time
                toolfis.write_summaries(csv_times, csv_summary, data_name_key, results, time_elapsed)


def eval_n_classifiers(folder_databases, databases, list_architectures, n_classifiers=[75]):
    folder_datetime = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(folder_datetime)
    for i in n_classifiers:  # i = 75
        folder_i_classifiers_evaluated = os.path.join(folder_datetime, str(i))
        os.makedirs(folder_i_classifiers_evaluated)
        eval_architectures(list_architectures, folder_i_classifiers_evaluated, folder_databases, databases, i)


def load_parameters_classifier():
    # Fuzzification parameters
    # categorical_bool_attributes = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # <<=====
    triangular_fuzzy_type = "tukey"  # "tukey", "normal"
    num_partitions_by_attribute = 3  # 3, 5, 7
    t_norm = "prod"  # "min", "prod"
    is_enable_negation = 1  # 0, 1

    # Formulation parameters
    size_premises = 2
    # Area filter parameters:
    criteria_area = "frequencia_relativa"  # "cardinalidade_relativa", "frequencia_relativa"
    area_threshold = 0.075
    # PCD filter parameter:
    is_enable_pcd = [1, 1]
    # Overlapping filter parameters:
    is_enable_overlapping = [1, 1]
    overlapping_threshold = 0.95

    # Association - Splitting
    method_splitting = "CD"  # "MQR", "PMQR", "CD", "PCD", "freq_max"

    #  Aggregation
    method_aggregation = "MQR"  # "MQR", "PMQR", "intMQR", "CD", "PCD", "max"

    # Grouping parameters
    parameters = [triangular_fuzzy_type, num_partitions_by_attribute, t_norm, is_enable_negation,
                  size_premises, criteria_area, area_threshold, is_enable_pcd,
                  is_enable_overlapping, overlapping_threshold,
                  method_splitting, method_aggregation]
    return parameters


def get_list_databases(folder_databases):
    files = os.listdir(folder_databases)
    return [i for i in files if i.endswith("_csv.zip")]


def main2():
    # BagFIS parameters
    n_classifiers = [25, 50]
    parameters = load_parameters_classifier()

    root_databases = os.path.dirname(os.path.realpath(__file__))
    databases = get_list_databases(root_databases)

    eval_n_classifiers(root_databases, databases, [parameters], n_classifiers)


def main():
    # BagFIS Parameters
    number_classifiers = 3
    parameters = load_parameters_classifier()

    root_databases = os.path.dirname(os.path.realpath(__file__))
    databases = get_list_databases(root_databases)

    root_outputs = root_databases
    csv_times, csv_summary = toolfis.create_csv_summaries(root_outputs)

    # Evaluate each database (zip file)
    for data in databases:
        is_successful, results_time = runBagFIS(root_databases, data, parameters, number_classifiers, root_outputs)
        if is_successful:
            data_name_key = data[0:-16]
            toolfis.write_summaries(csv_times, csv_summary, data_name_key, results_time[0], results_time[1])


if __name__ == '__main__':
    main()
