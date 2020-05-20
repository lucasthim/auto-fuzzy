__author__ = 'jparedes'

import os
from autoFIS_one_zip import cv_onezip
from databases_map import dictionary_data
import timeit


def define_parameters(database_name, pars):
    try:
        parameters = [dictionary_data(database_name)[0]] + pars  # addition of 2 list
    except KeyError:
        print "The database " + database_name + " was not found.\nIt was assumed that all attributes are numeric"
        parameters = [0]
    return parameters


def main():
    # -------------------------
    # Fuzzification parameters
    # -------------------------
    # categorical_bool_attributes = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # <<=====
    triangular_fuzzification_type = "tukey"  # "tukey", "normal"
    num_partitions_by_attribute = 3  # 3, 5, 7
    t_norm = "prod"  # "min", "prod"
    is_enable_negation = 1  # 0, 1

    # -------------------------
    # Formulation parameters
    # -------------------------
    ordem_max_premises = 2
    # Area filter parameters:
    criteria_area = "frequencia_relativa"  # "cardinalidade_relativa", "frequencia_relativa"
    area_threshold = 0.075
    # PCD filter parameter:
    is_enable_pcd = [0, 0]
    # Overlapping filter parameters:
    is_enable_overlapping = [1, 1]
    overlapping_threshold = 0.95

    # -------------------------
    # Association: ex Splitting
    # -------------------------
    method_association = "CD"  # "MQR", "PMQR", "CD", "PCD", "freq"

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
                  method_association, method_aggregation]

    # =============================================================================================== #
    # current_folder_path = os.path.dirname(os.path.realpath(__file__))
    current_folder_path = r"D:\Jorg\Disertacion\Datasets_keel\agrupando_por_tamano\magras(31)\now"
    print current_folder_path

    databases = []

    for archivo in os.listdir(current_folder_path):
        if archivo.endswith("_csv.zip"):
            databases.append(archivo)
    print databases
    # =============================================================================================== #

    file_times = open(os.path.join(current_folder_path, "Experimento_times.csv"), 'w')
    file_times.write("Dataset" + ", " + "Time(s)" + '\n')
    file_times.close()
    # Evaluate each database (zip file)
    for data in databases:
        t0 = timeit.default_timer()
        try:
            data_name_key = data[0:-16]
            parameters_database = define_parameters(data_name_key, parameters)
            achievement = cv_onezip(current_folder_path, data, parameters_database)
            if achievement == 0:
                raise ValueError("Problems in database: " + "<" + data + ">")
        except ValueError as e:
            print e
            achievement = 0
        tf = timeit.default_timer()

        if achievement:
            file_times = open(os.path.join(current_folder_path, "Experimento_times.csv"), 'a')
            file_times.write(data[:-16] + ', ' + str(tf - t0) + '\n')
            file_times.close()


if __name__ == '__main__':
    main()
