__author__ = 'jparedes'

import os
from autoFIS.lecture import Lecture
from autoFIS.fuzzification import Fuzzification
from autoFIS.formulation.formulation import Formulation
from autoFIS.association import Association
from autoFIS.aggregation import Aggregation
from autoFIS.decisions import Decisions
from autoFIS.evaluation import Evaluation

def autofis_onecv(file_csv, parameters):
    # General parameters
    cat_bool = parameters[0]
    fz_type = parameters[1]
    fz_number_partition = parameters[2]
    t_norm = parameters[3]
    is_enable_negation = parameters[4]
    maximum_size_of_premise = parameters[5]  # Ordem
    criteria_area = parameters[6]
    area_threshold = parameters[7]
    is_enable_pcd = parameters[8]
    is_enable_overlapping = parameters[9]
    overlapping_threshold = parameters[10]
    method_splitting = parameters[11]
    method_aggregation = parameters[12]

    # Gathering parameters
    # Formulation parameters:
    par_area = [criteria_area, area_threshold]
    par_over = [is_enable_overlapping, overlapping_threshold]
    par_pcd = [is_enable_pcd]

    # 1. Lecture
    lectura = Lecture()
    lectura.read_1file(file_csv)
    info = lectura.info_data()

    # 2. Fuzzification
    matrix_x = info[0].copy()
    f1 = Fuzzification(matrix_x, cat_bool)
    f1.build_uX(fz_type, fz_number_partition)
    if is_enable_negation == 1:
        f1.add_negation()

    # Getting train and test partitions # en este caso no hay teste =P
    #index_train = info[-1]
    ux_train = f1.uX
    #ux_test = f1.uX[index_train:, :]
    cbin_train = info[1]
    #cbin_test = info[1][index_train:, :]

    report = []  # Tengo duda si debe ser: report = ["\n"]

    try:
        # 3. Formulation
        num_premises_by_attribute = f1.size_attributes
        premises_by_attribute = f1.premises_attributes
        F2 = Formulation(ux_train, cbin_train, num_premises_by_attribute, premises_by_attribute)
        arbol = F2.gen_ARB(maximum_size_of_premise, t_norm, par_area, par_over, par_pcd)

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

        number_classes = cbin_train.shape[1]

        report.append("\nFormulation:\n-----------------")
        report.append("Elementos acorde a la profundidad " + str(len(arbol)) + " del arbol")
        for i in range(len(arbol)):
            report.append('Profundidad ' + str(i + 1) + ': ' + str(arbol[i][1].shape))
            # print 'Profundidad ' + str(i + 1) + ': ' + str(arbol[i][1].shape)

        # 4. Division
        F3 = Association(arbol, cbin_train)
        premises_by_class = F3.division(method_splitting)

        status = [0 if not i[0] else 1 for i in premises_by_class]
        if sum(status) != number_classes:
            raise ValueError("Error in Division Module. Some classes did not get premises. "
                             "Sorry, you can not continue in the next stage."
                             "\nTry to change the configuration")

        # 5. Aggregation:
        F4 = Aggregation(premises_by_class, cbin_train)
        output_aggregation = F4.aggregation(method_aggregation)

        premises_weights_names = output_aggregation[0]
        estimation_classes = output_aggregation[1]

        status = [0 if not i[0] else 1 for i in premises_weights_names]
        if sum(status) != number_classes:
            raise ValueError("Error in Aggregation Module. Some classes did not get premises. "
                             "Sorry, you can not continue in the next stage."
                             "\nTry to change the configuration")

        final_premises_classes = []
        report.append("\n\nPremises:\n=========")
        for i in range(len(premises_weights_names)):
            report.append("Premises of Class " + str(i) + ": " + str(premises_weights_names[i][0]))
            final_premises_classes.append(premises_weights_names[i][0])
            report.append("weights_" + str(i) + ": " + str(premises_weights_names[i][1].T))

        # 6. Decision:
        F5 = Decisions(estimation_classes, info[3])
        train_bin_prediction = F5.dec_max_pert()

        # 7. Evaluation
        F6 = Evaluation(premises_weights_names, final_premises_classes, info[3])
        metrics_train = F6.eval_train(cbin_train, train_bin_prediction)
        #metrics_test = F6.eval_test(cbin_test, ux_test, t_norm)

        report.append("\nEvaluation Training:\n---------------------------")
        report.append("Accuracy on train dataset: " + str(metrics_train[0]))
        report.append("AUC in train dataset: " + str(metrics_train[1]))
        report.append("Recall: " + str(metrics_train[3]))
        report.append('Confusion matrix:\n' + str(metrics_train[2]))

        # report.append("\nEvaluation Testing:\n---------------------------")
        # report.append("Accuracy on test dataset: " + str(metrics_test[0]))
        # report.append("AUC in test dataset: " + str(metrics_test[1]))
        # report.append("Recall: " + str(metrics_test[3]))
        # report.append("Confusion matrix:\n" + str(metrics_test[2]))

        # Metrics to eval: accuracy_test, auc_test,
        #                  [num_regras, total_rule_length, tamano_medio_das_regras]]
        # metricas = [1, [metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1], metrics_test[4]]]
        metricas = [1, [metrics_train[0], metrics_train[1]]]

    except ValueError as e:
        print e
        report = e  # .append("\n" + str(e))
        metricas = [0, "No se termino el proceso, se detuvo en algun etapa"]

    return report, metricas


def main():
    # -------------------------
    # Fuzzification parameters
    # -------------------------
    # categorical_bool_attributes = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # <<=====
    triangular_fuzzification_type = "tukey"  # "tukey", "normal"
    num_partitions_by_attribute = 5  # 3, 5, 7
    t_norm = "min"  # "min", "prod"
    is_enable_negation = 0 # 0, 1

    # -------------------------
    # Formulation parameters
    # -------------------------
    ordem_max_premises = 3
    # Area filter parameters:
    criteria_area = "frequencia_relativa"  # "cardinalidade_relativa", "frequencia_relativa"
    area_threshold = 0.02
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
    method_aggregation = "MQR"  # "MQR", "PMQR", "intMQR", "CD", "PCD", "max"

    # %%%%%%%%%%%%%%%%%%%%%%%%%%
    # %% Grouping parameters: %%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%
    parameters = [triangular_fuzzification_type, num_partitions_by_attribute, t_norm, is_enable_negation,
                  ordem_max_premises, criteria_area, area_threshold, is_enable_pcd,
                  is_enable_overlapping, overlapping_threshold,
                  method_splitting, method_aggregation]

    # =============================================================================================== #

    print 'Testando autoFis com 1 arquivo'
    current_folder_path = "C:\\Users\\jparedes\\Dropbox\Adriano-Jorge\\tarefa"
    file_path = "C:\\Users\\jparedes\\Dropbox\Adriano-Jorge\\tarefa\\planilhaRealMelautoFIS.csv"
    categorical_attributes = [5 * [0] + [1] + 2* [0] + [1] + 2 * [0]]

    parameters = categorical_attributes + parameters
    salida = autofis_onecv(file_path, parameters)

    print salida[0]

    if salida[1][0]:
        filename = os.path.join(current_folder_path, 'Reporte')
        target = open(filename, 'w')
        target.write('Parameters: ' + str(parameters))
        target.write('\n\n' + str(4*'===============================') + '\n\n')
        target.write('\n'.join(salida[0]))

        ac_train = salida[1][1][0]
        auc_train = salida[1][1][1]
        target.write('\n\n' + str(4*'===============================') + '\n\n')
        target.write('Accuracy training: ' + str(ac_train) + '\n')
        target.write('AUC training: ' + str(auc_train) + '\n')
        target.close()


if __name__ == '__main__':
    main()