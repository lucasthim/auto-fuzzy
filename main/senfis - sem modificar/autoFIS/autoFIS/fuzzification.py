__author__ = 'jparedes'

import numpy as np
import pandas as pd
from itertools import compress, chain
import scipy.sparse as sp

def trimf(x, params):
    """
    Triangular fuzzy operation in a vector
    :param x: Input column vector: array([[0.2],[0.9],[0.42],[0.74],[0.24],[0.28],[0.34]])
    :param params: 3 points os triangle shape: [0.1 0.4, 0.7]
    :return: Column vector: array([[0.], [0.], [0.8], [0.], [0.], [0.], [0.4]])
    """
    a = params[0]
    b = params[1]
    c = params[2]
    y = np.zeros(np.shape(x))  # Left and right shoulders (y = 0)

    # Left slope
    if a != b:
        index = np.logical_and(a < x, x < b)  # find(a < x & x < b)
        y[index] = (x[index] - a) / (b - a)

    # right slope
    if b != c:
        index = np.logical_and(b < x, x < c)  # find(b < x & x < c)
        y[index] = (c - x[index]) / (c - b)

    # Center (y = 1)
    index = x == b
    y[index] = 1
    return y


def trapmf(x, params):
    """
    Trapezoidal fuzzy operation
    :param x: Input column vector
    :param params: 4 points which define the trapezoidal
    :return: Output column vector
    """
    a, b, c, d = params
    y1 = np.zeros(np.shape(x))
    y2 = np.zeros(np.shape(x))

    # Compute y1
    index = x >= b
    if sum(index) != 0:  # ~isempty(index)
        y1[index] = 1.

    index = x < a
    if sum(index) != 0:  # ~isempty(index):
        y1[index] = 0.

    index = np.logical_and(a <= x, x < b)  # find(a <= x & x < b);
    ind = np.logical_and(sum(index) != 0, a != b)
    if ind:  # ~isempty(index) & a ~= b,
        y1[index] = (x[index] - a) / (b - a)

    # Compute y2
    index = x <= c
    if sum(index) != 0:
        y2[index] = 1.

    index = x > d
    if sum(index) != 0:
        y2[index] = 0.

    index = np.logical_and(c < x, x <= d)  # find(c < x & x <= d)
    ind = np.logical_and(sum(index) != 0, c != d)
    if ind:  # ~isempty(index) & c ~= d
        y2[index] = (d - x[index]) / (d - c)

    y = np.minimum(y1, y2)
    return y


def triangle_mb(y, tipo, n):
    """
    Aplication of 'n' triangular membership functions
    :param y: Attribute
    :param tipo: 'normal' or 'tukey'
    :param n: number of triangular membership functions
    :return:
    """
    if tipo == 'tukey':
        centro = np.percentile(y, np.linspace(0, 100, n).tolist())  # [0, 25, 50, 75, 100]
    else:  # 'normal'
        ymin = min(y)
        ymax = max(y)
        centro = np.linspace(ymin, ymax, n)

    ex_i = trapmf(y, [-np.inf, -np.inf, centro[0], centro[1]])
    ex_d = trapmf(y, [centro[n - 2], centro[n - 1], np.inf, np.inf])

    # Fuzzy sets
    muY = np.array(ex_i)

    for i in range(n - 2):
        aux = trimf(y, centro[i:i + n])
        muY = np.append(muY, aux, 1)

    muY = np.append(muY, ex_d, 1)

    return muY


def gather_columnspremises_by_attribute(lista, sizes_attributes):
    # lista:            [0,1,2,3...,7]
    # sizes_attributes: [3, 2, 3]
    # output:           [(0,1,2), (3,4), (5,6,7)]
    new_lista = []
    ref = 0
    for i in sizes_attributes:
        new_lista.append(tuple(lista[ref:ref+i]))
        ref += i
    return new_lista


class Fuzzification:
    def __init__(self, X, categorical_list_bool):
        self.X = X
        self.uX = []
        self.cat_list_bool = categorical_list_bool  # [0, 1, 0]
        self.num_prem_by_attribute = [0]                  # [3, 2, 3]
        self.premises_attributes = []               # [(0,1,2),(3,4),(5,6,7)]
        self.indexes_premises_contain_negation = []
        self.ref_attributes = range(len(categorical_list_bool))

    def build_uX(self, tipo, n):  # tipo con referencia a la fuzzificacion triangular: normal o tukey
        # Calculate 'uX' and 'size of attributes'
        list_uX = []
        size_attr = []
        MX = self.X

        if sum(self.cat_list_bool) != 0:
            for i in range(MX.shape[1]):
                if self.cat_list_bool[i] == 1:
                    attribute = pd.DataFrame(MX[:, [i]].tolist(), dtype="category")  # print attribute.describe()
                    aux = pd.get_dummies(attribute).values
                    if aux.shape[1] == 2:  # new IF
                        aux = np.delete(aux, 1, axis=1)
                    size_attr.append(aux.shape[1])
                else:
                    attribute = MX[:, [i]]
                    aux = triangle_mb(attribute, tipo, n)
                    size_attr.append(aux.shape[1])
                list_uX.append(aux)
        else:
            for i in range(MX.shape[1]):
                attribute = MX[:, [i]]
                aux = triangle_mb(attribute, tipo, n)
                list_uX.append(aux)
                size_attr.append(aux.shape[1])

        self.uX = np.hstack(list_uX)
        self.num_prem_by_attribute = size_attr
        number_columns = self.uX.shape[1]
        self.premises_attributes = gather_columnspremises_by_attribute(range(number_columns), size_attr)
        self.indexes_premises_contain_negation = number_columns * [0]

    def add_negation(self):
        num_attributes = len(self.cat_list_bool)
        # ref_attributes = range(num_attributes)
        num_col = sum(self.num_prem_by_attribute)  # number of columns, individual premises

        # attributes with more than 2 membership functions
        attrib_more_2fp = [0 if i < 3 else 1 for i in self.num_prem_by_attribute]
        index_premises_negation = [1 if (attrib_more_2fp[i] + 1 - self.cat_list_bool[i]) != 0
                                   else 0 for i in range(len(self.cat_list_bool))]
        attrib_survivors_negation = list(compress(range(num_attributes), index_premises_negation))

        premises_attrib_neg = gather_columnspremises_by_attribute(range(num_col, 2*num_col),
                                                                  list(pd.Series(self.num_prem_by_attribute)
                                                                       [attrib_survivors_negation]))  # Modified line
        premises_survivors_negation = list(compress(premises_attrib_neg, list(pd.Series(self.num_prem_by_attribute)
                                                                              [attrib_survivors_negation])))  # Modified line

        prem = [] # total premises (with negation) by attribute
        for i in range(num_attributes):
            prem_attr_i = self.premises_attributes[i]
            if i in attrib_survivors_negation:
                aux_index = attrib_survivors_negation.index(i)
                prem_attr_i += premises_survivors_negation[aux_index]
            prem.append(prem_attr_i)
        # self.uX = sp.csr_matrix(self.uX)
        # self.uX = sp.hstack((self.uX, 1. - self.uX), format='csr')
        prem_surv = pd.Series(self.premises_attributes)[attrib_survivors_negation]  # New line
        ind_neg = [i for sub in list(prem_surv) for i in sub]  # New line
        self.uX = np.concatenate((self.uX, 1. - self.uX[:, ind_neg]), axis=1)  # Modified line
        self.premises_attributes = prem[:]
        self.num_prem_by_attribute = [len(i) for i in prem]  # servira para el filtro de overlapping basicamente
        self.indexes_premises_contain_negation = index_premises_negation
        # self.ref_attributes = ref_attributes


def main():
    print ('Module 2 <<Fuzzification>>')


if __name__ == '__main__':
    main()