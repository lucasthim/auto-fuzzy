__author__ = 'jparedes'

import os
import zipfile
from .autoFIS_one_cv import autofis_onecv
from numpy import mean, std


def cv_onezip(path_databases, zip_file_name, parameters, path_output=0):
    # Este arquivo executa o processamento de uma base de dados, utilizando validação cruzada.
    # A base de dados (um arquivo zip) já é separada em 10 splits (em csv) para a validação cruzada.
    if path_output == 0:
        path_output = path_databases
    zipFilePath = os.path.join(path_databases, zip_file_name)

    # ==================================================================== #
    try:
        with zipfile.ZipFile(zipFilePath, 'r') as z:
            files_cv = z.namelist()

        number_files_zip = len(files_cv)
        if not (number_files_zip == 20 or number_files_zip == 10):
            raise ValueError("This module works with a zip file to 10cv or 5cv. "
                             "For this reason, it is expected 20 or 10 files inside the zip file")
        elif number_files_zip == 20:
            a = files_cv[2:] + files_cv[0:2]
        else:  # number_files_zip == 10
            a = files_cv

        list_train, list_test = a[::2], a[1::2]

        msg = []
        number_cv_pairs = int(number_files_zip / 2)
        ac_train = number_cv_pairs * [0]
        ac_test = number_cv_pairs * [0]
        auc_train = number_cv_pairs * [0]
        auc_test = number_cv_pairs * [0]

        num_rules = number_cv_pairs * [0]
        total_rule_length = number_cv_pairs * [0]

        for i in range(number_cv_pairs):
            print('Fold nº: ',i)
            train_file = list_train[i]
            test_file = list_test[i]

            message, indicators = autofis_onecv(zipFilePath, train_file, test_file, parameters)
            msg.append(message)

            if indicators[0] == 0:
                name_error = os.path.join(path_output, 'ERROR') + zip_file_name[:-13]
                fail_error = open(name_error, 'w')
                fail_error.write('Error in CV:' + str(i + 1))
                fail_error.write("\n" + message)
                fail_error.close()
                raise ValueError("Problem detected in CV " + str(i + 1))

            ac_train[i], ac_test[i] = indicators[1][0], indicators[1][1]
            auc_train[i], auc_test[i] = indicators[1][2], indicators[1][3]
            num_rules[i] = indicators[1][4][0]
            total_rule_length[i] = indicators[1][4][1]

        filename = os.path.join(path_output, 'Report of ') + zip_file_name[:-8]
        target = open(filename, 'w')
        target.write('Parameters: ' + str(parameters))
        for i2 in range(number_cv_pairs):
            target.write('\n\n' + str(4 * '===============================') + '\n\n')
            target.write('CV-' + str(i2 + 1) + '\n')
            target.write('\n'.join(msg[i2]))

        target.write('\n\n' + str(4 * '===============================') + '\n\n')
        target.write('Accuracy training: ' + str(mean(ac_train)) + ', ' + str(std(ac_train)) + '\n')
        target.write('Accuracy testing: ' + str(mean(ac_test)) + ', ' + str(std(ac_test)) + '\n')
        target.write('AUC training: ' + str(mean(auc_train)) + ', ' + str(std(auc_train)) + '\n')
        target.write('AUC testing: ' + str(mean(auc_test)) + ', ' + str(std(auc_test)) + '\n')
        target.write('Number of rules: ' + str(mean(num_rules)) + '\n')
        target.write('Total Rule Length: ' + str(mean(total_rule_length)))
        target.close()

        achievement = 1

        print ("win ", zip_file_name)

    except ValueError as e:
        print (e)
        achievement = 0

    return achievement


def main():
    currentfolder = os.path.dirname(os.path.realpath(__file__))
    filezip_name = "saheart-10-fold_csv.zip"

    # -------------------------
    # Fuzzification parameters
    # -------------------------
    categorical_bool_attributes = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    triangular_fuzzification_type = "normal"  # "tukey", "normal"
    num_partitions_by_attribute = 3
    t_norm = "min"  # "min", "prod"
    is_enable_negation = 0  # 0, 1

    # -------------------------
    # Formulation parameters
    # -------------------------
    ordem_max_premises = 2
    # Area filter parameters:
    criteria_area = "cardinalidade_relativa"  # "cardinalidade_relativa", "frequencia_relativa"
    area_threshold = 0.05
    # PCD filter parameter:
    is_enable_pcd = 1
    # Overlapping filter parameters:
    is_enable_overlapping = 1
    overlapping_threshold = 0.95

    # -------------------------
    # Splitting
    # -------------------------
    method_splitting = "MQR"  # "MQR", "PMQR", "CD", "PCD", "freq_max"

    # -------------------------
    # Aggregation
    # -------------------------
    method_aggregation = "MQR"  # "MQR", "PMQR", "CD", "PCD", "freq_max"

    # %%%%%%%%%%%%%%%%%%%%%%%%%%
    # %% Grouping parameters: %%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%
    parameters = [categorical_bool_attributes, triangular_fuzzification_type,
                  num_partitions_by_attribute, t_norm, is_enable_negation,
                  ordem_max_premises, criteria_area, area_threshold, is_enable_pcd,
                  is_enable_overlapping, overlapping_threshold,
                  method_splitting, method_aggregation]

    cv_onezip(currentfolder, filezip_name, parameters)


if __name__ == '__main__':
    main()
