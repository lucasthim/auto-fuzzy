from numpy import mean, std
import zipfile
import os
from itertools import product

from .autoFIS.autoFIS import utils_autofis as toolfis
from .RandomFISonecv_v3 import random_fis_one_cv, selection_criteria  # RandomFISonecv_wad, RandomFISonecv_v2
from .parameters_init import GlobalParameter


def packing_train_test_files(files_cv):
    if len(files_cv) == 20:  # number_files_zip
        a = sorted(files_cv)
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


def random_fis_one_zip(root, zip_file_name, parameters_classifiers, report_folder, clf_n, folder__tree_output,
                       data_name_key, time_ini):
    zipFilePath = os.path.join(root, zip_file_name)

    try:
        with zipfile.ZipFile(zipFilePath, 'r') as z:
            files_cv = z.namelist()

        number_files_zip = len(files_cv)
        if not (number_files_zip == 20 or number_files_zip == 10 or number_files_zip == 2):
            raise ValueError("This module works with a zip file to 10cv or 5cv. "
                             "For this reason, it is expected 20 or 10 files inside the zip file")
        number_cv_pairs = number_files_zip // 2
        list_train, list_test = packing_train_test_files(files_cv)

        msg = []
        output_container_results = container_results(number_cv_pairs)
        ac_train, ac_test, auc_train, auc_test, num_rules, rule_len, suc_classifiers = output_container_results
        cv_i = 0

        folder__tree_output = os.path.join(folder__tree_output, data_name_key)
        os.makedirs(folder__tree_output)

        parameters = parameters_classifiers[1:]
        print ("\n".join(map(str, parameters)))
        report_parameters = os.path.join(report_folder, 'Parameters_Classifiers.txt')
        with open(report_parameters, 'w') as r:
            r.write("\n".join(map(str, parameters)))

        param = GlobalParameter()

        for i in range(number_cv_pairs):

            train_file = list_train[i]
            test_file = list_test[i]

            ux_train, ux_test, cbin_train, cbin_test, freq_classes, parameters_classifiers, classifiers, cv_i,\
            outputs_tree_train, outputs_tree_train_bin, container_ac_train,\
            outputs_tree_test, outputs_tree_test_bin, partial_metrics_rules = \
                random_fis_one_cv(zipFilePath, train_file, test_file,
                                  parameters_classifiers, cv_i, clf_n)

            for (sel_method, sel_param, size_ensemble) in \
                    product(param.sel_method, param.sel_param, param.size_ensemble):

                message, indicators, cv_i, sel_successful_classifiers = \
                    selection_criteria(ux_train, ux_test, cbin_train, cbin_test, freq_classes,
                                       parameters_classifiers, classifiers, cv_i,
                                       outputs_tree_train, outputs_tree_train_bin, container_ac_train,
                                       outputs_tree_test, outputs_tree_test_bin, partial_metrics_rules,
                                       sel_method, sel_param, size_ensemble, folder__tree_output, time_ini)

                if indicators[0] == 0:
                    toolfis.create_error_file(root, zip_file_name, message, i)
                    raise ValueError("Problem detected in CV " + str(i + 1))

                # ac_train[i], ac_test[i] = indicators[1][0], indicators[1][1]
                # auc_train[i], auc_test[i] = indicators[1][2], indicators[1][3]
                # num_rules[i] = indicators[1][4][0]
                # rule_len[i] = indicators[1][4][1]
                # suc_classifiers[i] = indicators[2]

                sel_success = '_'.join(str(e) for e in sel_successful_classifiers)
                filename_report = os.path.join\
                    (report_folder, 'Report of ' + zip_file_name[:-8] + '_' + sel_success + '.txt')
                target = open(filename_report, 'a+')

                if i == 0:
                    aux_wr = parameters_classifiers[1:]
                    aux_wr.insert(0, 'Parameters Classifiers\n' + len('Parameters Classifiers')*'-')
                    summary_parameters = "\n".join(map(str, aux_wr))
                    cvs_info = [summary_parameters]
                else:
                    cvs_info = ["\n"]

                underline = "\n%s\n" % (110*"=")
                cv_summary = """    CV-{indice}:
                {cvv}""".format(indice=i + 1, cvv=message)
                cvs_info.append(cv_summary)

                # results = toolfis.summary_multiple(mean(suc_classifiers), [ac_train, auc_train],
                #                            [ac_test, auc_test], [num_rules, rule_len])
                # abstract = toolfis.template_results('Summary_CVs\n    -----------', results)
                # cvs_info.append(abstract)

                target.write(underline.join(cvs_info))
                target.close()

        achievement = 1
        print("win ", zip_file_name)

    except ValueError as e:
        print(e)
        achievement, ac_test, num_rules, suc_classifiers = 0, 0, 0, 0

    return achievement, (0, 0, 0, 0)

