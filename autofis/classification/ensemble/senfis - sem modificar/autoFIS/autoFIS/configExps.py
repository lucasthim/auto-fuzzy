__author__ = 'jparedes'

import os
import datetime
from expfunction import autofis_single_exp


def directorycreation(listexperiments):
    try:
        mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(mydir)
        evalexperiments(listexperiments, mydir)
    except OSError, e:
        if e.errno != 17:
            raise  # This was not a "directory exist" error..


def evalexperiments(experiments, path_databases, mydir):
    content = []
    for i in range(len(experiments)):
        aux = "Exp " + str(i + 1) + ":\n" + str(experiments[i]) + "\n"
        expfoldername = "Exp" + str(i + 1)
        content.append(aux)
        # Falta crear un folder por cada Exp
        try:
            expdir = os.path.join(mydir, expfoldername)
            os.makedirs(expdir)
            autofis_single_exp(experiments[i], path_databases, expdir)
        except OSError, e:
            if e.errno != 17:
                raise  # "directory exist"
    with open(os.path.join(mydir, "Resumo.txt"), 'w') as d:
        d.write('\n'.join(content))


def run_experiments_autofis(experiments, path_zip_datasets, path_output):
    # create a subfolder with the date in which experiments were run
    mydir = os.path.join(path_output, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
        evalexperiments(experiments, path_zip_datasets, mydir)
    except OSError, e:
        if e.errno != 17:
            raise  # This was not a "directory exist" error..


def main():
    # Configuration of Experiments
    # ............................................................ enablePCD, enableOverlap ......
    exp1 = [["tukey", 5, "prod", 1], [2, "cardinalidade_relativa", 0.1, [0, 0], [1, 1], 0.95], "CD", "MQR"]
    #exp2 = [["tukey", 3, "prod", 0], [2, "frequencia_relativa", 0.1, [0, 0], [1, 1], 0.95], "CD", "MQR"]
    exp3 = [["tukey", 3, "prod", 0], [2, "frequencia_relativa", 0.1, [0, 0], [1, 1], 0.95], "CD", "MQR"]
    #exp4 = [["normal", 3, "prod", 1], [2, "frequencia_relativa", 0.05, [0, 0], [1, 1], 0.95], "MQR", "MQR"]
    # exp5 = [["tukey", 3, "prod", 1], [2, "cardinalidade_relativa", 0.025, [0, 0], [1, 1], 0.95], "MQR",
    #        "MQR"]  # cuidado
    # exp6 = [["tukey", 3, "prod", 0], [4, "cardinalidade_relativa", 0.025, [0, 0], [1, 1], 0.95], "MQR", "MQR"]
    experiments = [exp3] #, exp2, exp3, exp4]

    path_datasets = r"C:\Users\OSCAR\Desktop\autoFIS\test\datas"
    path_output = r"C:\Users\OSCAR\Desktop\autoFIS\test\datas"  # "D:\\Jorg\\Disertacion"  where reports will be stored by experiment

    run_experiments_autofis(experiments, path_datasets, path_output)  # el path_output_reports es opcional


if __name__ == '__main__':
    main()
