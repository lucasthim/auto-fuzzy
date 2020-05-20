__author__ = 'jparedes'

import numpy as np
import pandas as pd
from itertools import compress, chain
import scipy.sparse as sp

class Fuzzification:

    '''
    Fuzzification module of a Fuzzy Inference System

    This class is responsible for building the membership functions for an attribute
    '''
    
    def __init__(self, X, categorical_attributes_mask = None):
        self.X = X
        self.uX = []
        self.categorical_attributes_mask = categorical_attributes_mask  # [0, 1, 0]
        self.num_of_premises_by_attribute = [0]                  # [3, 2, 3]
        self.attribute_premises = []               # [(0,1,2),(3,4),(5,6,7)]
        self.indexes_premises_contain_negation = []
        self.ref_attributes = range(len(categorical_attributes_mask))

    def build_uX(self, triangle_format, n):  
        # triangle_format con referencia a la fuzzificacion triangular: normal o tukey
        # Calculate 'uX' and 'size of attributes'
        list_uX = []
        size_attr = []
        MX = self.X
        if self.categorical_attributes_mask:
            for i in range(MX.shape[1]):
                if self.categorical_attributes_mask[i] == 1:
                    attribute = pd.DataFrame(MX[:, [i]].tolist(), dtype="category")  # print attribute.describe()
                    aux = pd.get_dummies(attribute).values
                    if aux.shape[1] == 2:  # new IF
                        aux = np.delete(aux, 1, axis=1)
                    size_attr.append(aux.shape[1])
                else:
                    attribute = MX[:, [i]]
                    aux = triangle_mb(attribute, triangle_format, n)
                    size_attr.append(aux.shape[1])
                list_uX.append(aux)
        else:
            for i in range(MX.shape[1]):
                attribute = MX[:, [i]]
                aux = triangle_mb(attribute, triangle_format, n)
                list_uX.append(aux)
                size_attr.append(aux.shape[1])

        self.uX = np.hstack(list_uX)
        self.num_of_premises_by_attribute = size_attr
        number_columns = self.uX.shape[1]
        self.attribute_premises = self.gather_columnspremises_by_attribute(range(number_columns), size_attr)
        self.indexes_premises_contain_negation = number_columns * [0]


    def add_negation(self):
        num_attributes = len(self.categorical_attributes_mask)
        # ref_attributes = range(num_attributes)
        num_col = sum(self.num_of_premises_by_attribute)  # number of columns, individual premises

        # attributes with more than 2 membership functions
        attrib_more_2fp = [0 if i < 3 else 1 for i in self.num_of_premises_by_attribute]
        index_premises_negation = [1 if (attrib_more_2fp[i] + 1 - self.categorical_attributes_mask[i]) != 0
                                   else 0 for i in range(len(self.categorical_attributes_mask))]
        attrib_survivors_negation = list(compress(range(num_attributes), index_premises_negation))

        premises_attrib_neg = self.gather_columnspremises_by_attribute(range(num_col, 2*num_col),
                                                                  list(pd.Series(self.num_of_premises_by_attribute)
                                                                       [attrib_survivors_negation]))  # Modified line
        premises_survivors_negation = list(compress(premises_attrib_neg, list(pd.Series(self.num_of_premises_by_attribute)
                                                                              [attrib_survivors_negation])))  # Modified line

        prem = [] # total premises (with negation) by attribute
        for i in range(num_attributes):
            prem_attr_i = self.attribute_premises[i]
            if i in attrib_survivors_negation:
                aux_index = attrib_survivors_negation.index(i)
                prem_attr_i += premises_survivors_negation[aux_index]
            prem.append(prem_attr_i)
        
        prem_surv = pd.Series(self.attribute_premises)[attrib_survivors_negation]  # New line
        ind_neg = [i for sub in list(prem_surv) for i in sub]  # New line
        self.uX = np.concatenate((self.uX, 1. - self.uX[:, ind_neg]), axis=1)  # Modified line
        self.attribute_premises = prem[:]
        self.num_of_premises_by_attribute = [len(i) for i in prem]  # servira para el filtro de overlapping basicamente
        self.indexes_premises_contain_negation = index_premises_negation


    def trimf(self,x, triangle_points):
        """
        Triangular fuzzy operation in a vector
        
        Parameters:
            x: Input column vector: [[0.2],[0.9],[0.42],[0.74],[0.24],[0.28],[0.34]]
            
            triangle_points: array with containing support and peak of triangle: [0.1 0.4, 0.7]
        
        Return: Column vector: array([[0.], [0.], [0.8], [0.], [0.], [0.], [0.4]])
        """
        a = triangle_points[0]
        b = triangle_points[1]
        c = triangle_points[2]
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


    def trapmf(self,x, trapz_points):

        """
        Trapezoidal fuzzy operation
        
        Parameters:
            
            x: Input column vector;

            trapz_points: 4 points which define the trapezoid: [0.1, 0.4, 0.6, 0.7]
        
        return: Output column vector
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


    def triangle_mb(self,y, triangle_format = 'normal', n_membership_functions = 3):

        """
        Aplication of 'n' triangular membership functions
        
        Parameters:
            
            y: Attribute
            
            triangle_format: 'normal' or 'tukey'
            
            n_membership_functions: number of triangular membership functions
            
        Return: array with membership functions
        """

        if triangle_format == 'tukey':
            center = np.percentile(y, np.linspace(0, 100, n_membership_functions).tolist())
        else:
            ymin = min(y)
            ymax = max(y)
            center = np.linspace(ymin, ymax, n)

        membership_far_left = self.trapmf(y, [-np.inf, -np.inf, center[0], center[1]])
        membership_far_right = self.trapmf(y, [center[n - 2], center[n - 1], np.inf, np.inf])

        fuzzy_sets = np.array(membership_far_left)
        for i in range(n_membership_functions - 2):
            aux = self.trimf(y, center[i:i + n])
            fuzzy_sets = np.append(fuzzy_sets, aux, 1)

        fuzzy_sets = np.append(fuzzy_sets, membership_far_right, 1)

        return fuzzy_sets


    def gather_columnspremises_by_attribute(self, premises_list, sizes_attributes):
        
        """
        Gather columnspremises by each attribute
        
        Parameters:
            
            premises_list: Array with number of premises [0,1,2,3...,7]
            
            sizes_attributes: Array with membership functions per attribute [3, 2, 3]
            
            n_membership_functions: number of triangular membership functions
            
        Return: array with each premise grouped in a tuple [(0,1,2), (3,4), (5,6,7)]
        """

        new_premises_list = []
        ref = 0
        for i in sizes_attributes:
            new_premises_list.append(tuple(premises_list[ref:ref+i]))
            ref += i
        return new_premises_list


def main():
    print ('Module 2 <<Fuzzification>>')

if __name__ == '__main__':
    main()