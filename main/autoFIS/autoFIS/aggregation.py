__author__ = 'jparedes'

# -*- coding: utf-8 -*-
import numpy as np
from itertools import compress
from sklearn import linear_model  # to use LogisticRegression
from .auxfunc import calculate_cd_transpose, iter_beta, int_premisas


class Aggregation:

    def __init__(self, association_rules, target_class_one_hot):
        
        """
        :param association_rules:
                [[premisas_c1 uC1], [premisas_c2 uC2], ....]
        """

        self.association_rules = association_rules
        self.target_class_one_hot = target_class_one_hot
        self.association_method_mapper = {"max": self.max_method,
                                 "LinearModel": self.linear,
                                 "LogRegression": self.logregression,
                                 "MQR": self.mqr,
                                 "intMQR": self.int_mqr}

    def aggregate_rules(self, method='MQR', tnorm_int_mqr='min'):
        data_to_aggregate = [self.association_rules, self.target_class_one_hot]
        if method == "intMQR":
            data_to_aggregate += [2, tnorm_int_mqr]

        # Choose a method and return the estimation and summary of model
        return self.association_method_mapper[method](data_to_aggregate)

    @staticmethod
    def linear(data_to_aggregate):
        indexes_and_premises_by_class, classes_bin = data_to_aggregate
        number_of_classes = len(indexes_and_premises_by_class)

        summary_classes = []
        model_per_class = []
        linear_estimation = 1. * classes_bin.copy()

        for i in range(number_of_classes):
            indexes = indexes_and_premises_by_class[i][0]
            u_premises = indexes_and_premises_by_class[i][1]
            class_bin = classes_bin[:, [i]]

            betas = calculate_cd_transpose(u_premises, class_bin)
            betas = betas / betas.sum(axis=1)

            logica = betas > 0
            new_indexes = list(compress(indexes, logica[0]))
            weights = betas[:, logica[0]]

            class_linear_output = np.dot(u_premises[:, logica[0]], weights.T)
            linear_estimation[:, [i]] = class_linear_output

            model_per_class.append([new_indexes, weights, 'lin_regre'])

        summary_classes.append(model_per_class)
        summary_classes.append(linear_estimation)

        return summary_classes

    @staticmethod
    def logregression(data_to_aggregate):
        indexes_and_premises_by_class, classes_bin = data_to_aggregate
        number_of_classes = classes_bin.shape[1]  # len(indexes_and_premises_by_class)

        summary_classes = []
        model_per_class = []
        log_reg_estimation = 1. * classes_bin.copy()
        for i in range(number_of_classes):
            indexes = indexes_and_premises_by_class[i][0]
            u_premises = indexes_and_premises_by_class[i][1]
            class_bin = classes_bin[:, i]  # classes_bin[:,[i]]

            model_i = linear_model.LogisticRegression(C=1e5)
            model_i.fit(u_premises, class_bin)
            log_reg_estimation[:, i] = model_i.predict(u_premises)
            model_per_class.append([indexes, model_i, 'log_regre'])

        summary_classes.append(model_per_class)
        summary_classes.append(log_reg_estimation)
        return summary_classes

    @staticmethod
    def max_method(data_to_aggregate):
        indexes_and_premises_by_class, classes_bin = data_to_aggregate
        number_of_classes = len(indexes_and_premises_by_class)
        model_per_class = []
        max_estimation = 1. * classes_bin.copy()
        for i in range(number_of_classes):
            indexes = indexes_and_premises_by_class[i][0]
            u_premises = indexes_and_premises_by_class[i][1]
            max_estimation[:, i] = u_premises.max(axis=1)
            # Falta calcular los new_indexes
            model_per_class.append([indexes, np.ones((1, 1)), 'max'])

        return [model_per_class, max_estimation]  # summary of all classes

    @staticmethod
    def mqr(data_to_aggregate):
        indexes_and_premises_by_class, classes_bin = data_to_aggregate
        number_of_classes = len(indexes_and_premises_by_class)

        summary_classes = []
        model_per_class = []
        mqr_estimation = 1. * classes_bin.copy()

        for i in range(number_of_classes):
            indexes = indexes_and_premises_by_class[i][0]
            u_premises = indexes_and_premises_by_class[i][1]
            class_bin = classes_bin[:, [i]]

            betas = iter_beta(u_premises, class_bin)

            logica = betas > 0  # vector logico columna
            new_indexes = list(compress(indexes, logica.T[0]))
            weights = betas[logica.T[0], :]

            mqr_estimation[:, [i]] = np.dot(u_premises[:, logica.T[0]], weights)

            model_per_class.append([new_indexes, weights, 'MQR'])

        summary_classes.append(model_per_class)
        summary_classes.append(mqr_estimation)
        return summary_classes

    @staticmethod
    def int_mqr(data_to_aggregate):
        indexes_and_premises_by_class, classes_bin, m, tnorm = data_to_aggregate
        number_of_classes = len(indexes_and_premises_by_class)

        summary_classes = []
        model_per_class = []
        int_mqr_estimation = 1. * classes_bin.copy()

        for i in range(number_of_classes):
            class_bin = classes_bin[:, [i]]

            indexes_o = indexes_and_premises_by_class[i][0]
            uo_premises = indexes_and_premises_by_class[i][1]
            indexes, u_premises = int_premisas(indexes_o, uo_premises, m, tnorm)

            betas = iter_beta(u_premises, class_bin)

            logica = betas > 0  # vector logico columna
            new_indexes = list(compress(indexes, logica.T[0]))
            weights = betas[logica.T[0], :]

            int_mqr_estimation[:, [i]] = np.dot(u_premises[:, logica.T[0]], weights)

            model_per_class.append([new_indexes, weights, 'intMQR'])

        summary_classes.append(model_per_class)
        summary_classes.append(int_mqr_estimation)
        return summary_classes


def main():
    pass
    # p1 = [(2,), (5, 6), (4, 8), (7,), (9, 11)]
    # a1 = np.array([[0.1000, 0.0300, 0.2000, 0.1300, 0.0060],
    #                [0.0100, 0.0020, 0.0100, 0.4000, 0.0700],
    #                [0.5000, 0.0500, 0.0200, 0.0040, 0.0300],
    #                [0.0200, 0.0500, 0.0600, 0.3000, 0.7000],
    #                [0.3000, 0.1000, 0.0200, 0.2000, 0.3000],
    #                [0.0600, 0.2000, 0.0700, 0.0900, 0.4000],
    #                [0.2000, 0.0600, 0.7000, 0.2500, 0.2300]])

    # p2 = [(7,), (12, 6), (3, 4), (1,), (2, 11), (13,), (15, 17), (20, 25), (32, 35), (39,)]
    # a2 = np.array([[0.0200, 0.1000, 0.0200, 0.0300, 0.0500, 0.0400, 0.1500, 0.3300, 0.1200, 0.2400],
    #                [0.1000, 0.0200, 0.0240, 0.0670, 0.0780, 0.0450, 0.0340, 0.0670, 0.0100, 0.0300],
    #                [0.1200, 0.4230, 0.1420, 0.7300, 0.4500, 0.0400, 0.0500, 0.0300, 0.0200, 0.1740],
    #                [0.0100, 0.0020, 0.0154, 0.0370, 0.1800, 0.4500, 0.0400, 0.7000, 0.0100, 0.1000],
    #                [0.3000, 0.0120, 0.0154, 0.3700, 0.8100, 0.1500, 0.4400, 0.4700, 0.2310, 0.7810],
    #                [0.0400, 0.0780, 0.0154, 0.0370, 0.1780, 0.4500, 0.2040, 0.8070, 0.0410, 0.3100],
    #                [0.0800, 0.6200, 0.0154, 0.7000, 0.1800, 0.1060, 0.0140, 0.1070, 0.1110, 0.2210]])

    # target_class_one_hot = np.array([[1, 0],
    #                  [1, 0],
    #                  [0, 1],
    #                  [1, 0],
    #                  [0, 1],
    #                  [0, 1],
    #                  [1, 0]])

    # libom = [[p1, a1], [p2, a2]]
    # output = aggregate_rules(libom, target_class_one_hot)

    # print (output.aggregate_rules('linear'))
    # print (output.aggregate_rules('LogRegression'))
    # print (output.aggregate_rules('MQR'))
    # print (output.aggregate_rules('intMQR', 'min'))
    # print (output.aggregate_rules('max'))
    # print (5)

if __name__ == '__main__':
    main()

'''     Initial code in lineal_model using get_CD
betas = get_CD(u_premises, class_bin)
betas = betas / betas.sum(axis=0)

# Reduction of premises (in case zero weights are detected)
logica = betas > 0
new_indexes = [indexes[j] for j in range(len(indexes)) if logica[j]]
weigths = betas[logica.T[0], :]

pM = np.dot(u_premises[:, logica.T[0]], weigths)
M.append(pM)

model_per_classe.append([new_indexes, weigths.T, 'lin_regre'])
'''