import glob
import os
import re
import numpy as np
import autoFIS.autoFIS.utils_autofis as toolfis


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


def main():

    directory = os.getcwd()
    os.chdir(f'{directory}/2019-01-31_06-42-36/15')
    for filename in glob.glob('Report*.txt'):

        with open(filename, 'r') as file_tmp:

            suc_classifiers, ac_train, auc_train, ac_test, auc_test, num_rules, rule_len = [], [], [], [], [], [], []
            for line in file_tmp.read().splitlines():

                if "Successful classifiers: " in line:
                    suc_classifiers.append(re.search(r'\[(.*?)\]', line).group(1))
                elif "Accuracy training: " in line:
                    ac_train.append(float(re.findall("\d+\.\d+", line)[0]))
                elif "AUC Training: " in line:
                    if len(re.findall("\d+\.\d+", line)) == 0:
                        auc_train.append(0)
                    else:
                        auc_train.append(float(re.findall("\d+\.\d+", line)[0]))
                elif "Accuracy Testing: " in line:
                    ac_test.append(float(re.findall("\d+\.\d+", line)[0]))
                elif "AUC Testing: " in line:
                    if len(re.findall("\d+\.\d+", line)) == 0:
                        auc_test.append(0)
                    else:                        
                        auc_test.append(float(re.findall("\d+\.\d+", line)[0]))
                elif "Number of Rules: " in line:
                    num_rules.append(int(re.findall("\d+", line)[0]))
                elif "Total Rule Length: " in line:
                    rule_len.append(int(re.findall("\d+", line)[0]))

            results = toolfis.summary_multiple(suc_classifiers[1], [ac_train, auc_train],
                                               [ac_test, auc_test], [num_rules, rule_len])

        abstract = toolfis.template_results('Summary_CVs\n    -----------', results)
        underline = "\n%s\n" % (110 * "=")

        file = open(filename, 'a+')
        file.write(underline)
        file.write(abstract)

        file.close()

main()
