__author__ = 'jparedes'


import numpy as np
import pandas as pd
from itertools import compress, chain
import scipy.sparse as sp

# This class is deprecated
class Fuzzification:

    '''

    Fuzzification module of a Fuzzy Inference System

    This class is responsible for building the membership functions for an attribute.

    '''
    
    def __init__(self,X,categorical_attributes_mask = [], fuzzy_sets_by_attribute = 3, triangle_format = 'normal', enable_negation = False):
        
        
        self.X = X #array([[1,2,3,4],[2,3,4,5],...])
        self.categorical_attributes_mask = categorical_attributes_mask or X.shape ==(X.shape[1] * [False]) 
        self.fuzzy_sets_by_attribute = fuzzy_sets_by_attribute
        self.triangle_format = triangle_format
        self.enable_negation = enable_negation

        self.uX = []
        self.num_of_antecedents_by_attribute = []  # [3, 2, 3]
        self.antecedents_by_attribute = []                 # [(0,1,2),(3,4),(5,6,7)]
        self.attributes_negation_mask = []              # [True, False, True]
        self.aux = None

    def build_membership_functions(self):  
        
        '''
        Build membership functions for attributes
        
        Parameters:
            X: data to fuzzify 
            categorical_attributes_mask: array of booleans indicating which attributes are categorical
            triangle_format: 'normal' or 'tukey'
            fuzzy_sets_by_attribute: number of membership functions per attribute. Generally 3,5 or 7
            enable_negation: boolean to enable creation of membership function negations. 
        '''

        list_uX = []
        size_attr = []
        for attr in range(self.X.shape[1]):
            if self.categorical_attributes_mask[attr]:
                attribute = pd.DataFrame(self.X[:, [attr]].tolist(), dtype="category")  # print attribute.describe()
                aux = pd.get_dummies(attribute).values
                if aux.shape[1] == 2:
                    aux = np.delete(aux, 1, axis=1)
            else:
                attribute = self.X[:, [attr]]
                aux = self.triangle_mb(attribute, self.triangle_format, self.fuzzy_sets_by_attribute)

            list_uX.append(aux)
            size_attr.append(aux.shape[1])

        # list_uX é um tensor: atributos x samples x sets
        self.uX = np.hstack(list_uX)
        self.num_of_antecedents_by_attribute = size_attr
        self.antecedents_by_attribute = self.gather_columnspremises_by_attribute(range(self.uX.shape[1]), size_attr)
        self.attributes_negation_mask = self.X.shape[1] * [False]

        if self.enable_negation:
            self.add_negation()


    def add_negation(self):

        '''
        Build membership functions  with negation for attributes
        Parameters:
            X: data to fuzzify 
            categorical_attributes_mask: array of booleans indicating which attributes are categorical
            fuzzy_sets_by_attribute: number of membership functions per attribute. Generally 3,5 or 7
            enable_negation: boolean to enable creation of membership function negations. 
        '''
        
        num_attributes = len(self.categorical_attributes_mask)
        num_col = sum(self.num_of_antecedents_by_attribute) 

        attr_with_more_than_2_mf = [False if i < 3 else True for i in self.num_of_antecedents_by_attribute]
        index_premises_negation = [True if (attr_with_more_than_2_mf[i] + 1 - self.categorical_attributes_mask[i]) != 0
                                   else False for i in range(len(self.categorical_attributes_mask))]
        attrib_survivors_negation = list(compress(range(num_attributes), index_premises_negation))

        premises_attrib_neg = self.gather_columnspremises_by_attribute(range(num_col, 2*num_col),
                                                                  list(pd.Series(self.num_of_antecedents_by_attribute)
                                                                       [attrib_survivors_negation]))  # Modified line
        premises_survivors_negation = list(compress(premises_attrib_neg, list(pd.Series(self.num_of_antecedents_by_attribute)
                                                                              [attrib_survivors_negation])))  # Modified line

        total_premises = []
        for i in range(num_attributes):
            prem_attr_i = self.antecedents_by_attribute[i]
            if i in attrib_survivors_negation:
                aux_index = attrib_survivors_negation.index(i)
                prem_attr_i += premises_survivors_negation[aux_index]
            total_premises.append(prem_attr_i)
        
        prem_surv = pd.Series(self.antecedents_by_attribute)[attrib_survivors_negation] 
        ind_neg = [i for sub in list(prem_surv) for i in sub] 
        
        self.uX = np.concatenate((self.uX, 1. - self.uX[:, ind_neg]), axis=1)
        self.antecedents_by_attribute = total_premises[:]
        self.num_of_antecedents_by_attribute = [len(i) for i in total_premises]
        self.attributes_negation_mask = index_premises_negation
        
        self.index_premises_negation = index_premises_negation
        self.attrib_survivors_negation = attrib_survivors_negation
        self.premises_attrib_neg = premises_attrib_neg
        self.premises_survivors_negation = premises_survivors_negation
        self.total_premises = total_premises
        self.ind_neg = ind_neg
        self.prem_surv = prem_surv

        # return self.uX

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

            trapz_points: 4 points define the trapezoid: [0.1, 0.4, 0.6, 0.7]
        
        return: Output column vector
        """

        a, b, c, d = trapz_points
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


    def triangle_mb(self,y, triangle_format = 'normal', n_fuzzy_sets = 3):

        """
        Build triangular membership functions
        
        Parameters:
            
            y: Attribute
            triangle_format: 'normal' or 'tukey'
            n_fuzzy_sets: number of fuzzy sets

        Return: array with membership functions
        """

        if triangle_format == 'tukey':
            center = np.percentile(y, np.linspace(0, 100, n_fuzzy_sets).tolist())
            # print(center)
        else:
            ymin = min(y)
            ymax = max(y)
            center = np.linspace(ymin, ymax, n_fuzzy_sets)
            # print('min: ',ymin,'; max: ',ymax,' centers: ',center)
        membership_far_left = self.trapmf(y, [-np.inf, -np.inf, center[0], center[1]])
        membership_far_right = self.trapmf(y, [center[n_fuzzy_sets - 2], center[n_fuzzy_sets - 1], np.inf, np.inf])

        fuzzy_sets = np.array(membership_far_left)
        for i in range(n_fuzzy_sets - 2):
            aux = self.trimf(y, center[i:i + n_fuzzy_sets])
            fuzzy_sets = np.append(fuzzy_sets, aux, 1)

        fuzzy_sets = np.append(fuzzy_sets, membership_far_right, 1)

        return fuzzy_sets


    def gather_columnspremises_by_attribute(self, premises_list, sizes_attributes):
        
        """
        Gather columnspremises by each attribute
        
        Parameters:
            
            premises_list: Array with number of premises [0,1,2,3...,7]
            
            sizes_attributes: Array with membership functions per attribute [3, 2, 3]
            
            n_fuzzy_sets: number of fuzzy sets
            
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