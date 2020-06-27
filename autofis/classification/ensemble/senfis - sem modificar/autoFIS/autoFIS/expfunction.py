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


def autofis_single_exp(parameters, path_datasets, path_reports):
    # Example parameters:
    # Exp = [fuzzification, formulation, association, aggregation]
    # Exp = [["normal",3,"prod",1], [2,"cardinalidade_relativa", 0.075, [1,1], [1,1], 0.95], "CD", "MQR"]

    triangular_fuzzification_type, num_partitions_by_attribute, t_norm, is_enable_negation = parameters[0]
    ordem_max_premises, criteria_area, area_threshold,\
        is_enable_pcd, is_enable_overlapping, overlapping_threshold = parameters[1]

    method_splitting = parameters[2]
    method_aggregation = parameters[3]

    parameters = [triangular_fuzzification_type, num_partitions_by_attribute, t_norm, is_enable_negation,
                  ordem_max_premises, criteria_area, area_threshold, is_enable_pcd,
                  is_enable_overlapping, overlapping_threshold,
                  method_splitting, method_aggregation]

    # =============================================================================================== #
    # current_folder_path = os.path.dirname(os.path.realpath(__file__))
    # path_datasets = "D:\\Jorg\\Disertacion\\Datasets_keel\\agrupando_por_tamano\\magras(31)\\now"
    # print path_datasets

    databases = []

    for archivo in os.listdir(path_datasets):
        if archivo.endswith("_csv.zip"):
            databases.append(archivo)
    print databases
    # =============================================================================================== #

    file_times = open(os.path.join(path_reports, "Experimento_times.csv"), 'w')
    file_times.write("Dataset" + ", " + "Time(s)" + '\n')
    file_times.close()

    # Evaluate each database (zip file)
    for data in databases:
        t0 = timeit.default_timer()
        try:
            data_name_key = data[0:-16]
            parameters_database = define_parameters(data_name_key, parameters)
            achievement = cv_onezip(path_datasets, data, parameters_database, path_reports)
            if achievement == 0:
                raise ValueError("Problems in database: " + "<" + data + ">")
        except ValueError as e:
            print e
        tf = timeit.default_timer()

        if achievement == 1:
            file_times = open(os.path.join(path_reports, "Experimento_times.csv"), 'a')
            file_times.write(data[:-16] + ', ' + str(tf-t0) + '\n')
            file_times.close()

def main():
    print 5

if __name__ == '__main__':
    main()
