__author__ = 'jparedes'

import utils_autofis as toolfis
from formulation.formulation import Formulation
from association import Association
from aggregation import Aggregation
from decisions import Decisions
from evaluation import Evaluation


def autofis_onecv(file_zip, file_train, file_test, parameters):
    # General parameters
    t_norm = parameters[3]
    max_size_of_premise = parameters[5]
    association_method = parameters[11]
    aggregation_method = parameters[12]

    # Gathering parameters
    # Formulation parameters:
    par_area, par_over, par_pcd = toolfis.get_formulation_parameters(parameters)

    # 1. Lecture & Fuzzification
    out1 = toolfis.lecture_fuz_one_cv(file_zip, file_train, file_test, parameters)
    ux_train, cbin_train = out1[0]
    ux_test, cbin_test = out1[1]
    num_premises_by_attribute, premises_by_attribute, ref_attributes, premises_contain_negation = out1[2]
    freq_classes = out1[3]

    report = []  # To save our results

    try:
        # 3. Formulation
        f2 = Formulation(ux_train, cbin_train, ref_attributes, premises_by_attribute,
                         num_premises_by_attribute, premises_contain_negation)
        # Inputs given by user
        arbol = f2.gen_ARB(max_size_of_premise, t_norm, par_area, par_over, par_pcd)

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

        # 4. Association: ex-Division
        f3 = Association(arbol, cbin_train)
        premises_ux_by_class = f3.division(association_method)

        status = [0 if not i[0] else 1 for i in premises_ux_by_class]
        if sum(status) != number_classes:
            raise ValueError("Error in Division Module. Some classes did not get premises. "
                             "Sorry, you can not continue in the next stage."
                             "\nTry to change the configuration")

        # 5. Aggregation:
        f4 = Aggregation(premises_ux_by_class, cbin_train)
        output_aggregation = f4.aggregation(aggregation_method)

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
        f5 = Decisions(estimation_classes, freq_classes)
        train_bin_prediction = f5.dec_max_pert()

        # 7. Evaluation
        f6 = Evaluation(premises_weights_names, final_premises_classes, freq_classes)
        metrics_train = f6.eval_train(cbin_train, train_bin_prediction)
        metrics_test = f6.eval_test(cbin_test, ux_test, t_norm)

        report.append("\nEvaluation Training:\n---------------------------")
        report.append("Accuracy on train dataset: " + str(metrics_train[0]))
        report.append("AUC in train dataset: " + str(metrics_train[1]))
        report.append("Recall: " + str(metrics_train[3]))
        report.append('Confusion matrix:\n' + str(metrics_train[2]))

        report.append("\nEvaluation Testing:\n---------------------------")
        report.append("Accuracy on test dataset: " + str(metrics_test[0]))
        report.append("AUC in test dataset: " + str(metrics_test[1]))
        report.append("Recall: " + str(metrics_test[3]))
        report.append("Confusion matrix:\n" + str(metrics_test[2]))

        # Metrics to eval: accuracy_test, auc_test,
        #                  [num_regras, total_rule_length, tamano_medio_das_regras]]
        metricas = [1, [metrics_train[0], metrics_test[0], metrics_train[1], metrics_test[1], metrics_test[4]]]

    except ValueError as e:
        print e
        report = e  # .append("\n" + str(e))
        metricas = [0, "No se termino el proceso, se detuvo en algun etapa"]

    return report, metricas


def main():
    filezip_name = "D:\\Jorg\Projects\\autoFIS\\test\\datas" + '\\' + 'saheart-10-fold_csv.zip'
    train_file = "saheart-10-7tra.csv"
    test_file = "saheart-10-7tst.csv"

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
    # Association
    # -------------------------
    method_association = "MQR"  # "MQR", "PMQR", "CD", "PCD", "freq_max"

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
                  method_association, method_aggregation]

    result_1cv = autofis_onecv(filezip_name, train_file, test_file, parameters)
    print result_1cv[1]


if __name__ == '__main__':
    main()