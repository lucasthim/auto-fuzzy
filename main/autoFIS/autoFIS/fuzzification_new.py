
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
from itertools import compress, chain
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class FuzzificationNew(BaseEstimator,TransformerMixin):

    '''

    Fuzzification module of a Fuzzy Inference System

    This class is responsible for building the membership functions for an attribute.

    '''
    
    def __init__(self, n_fuzzy_sets = 3, triangle_format = 'normal'):
        
        
        self.n_fuzzy_sets = n_fuzzy_sets
        self.triangle_format = triangle_format

        self.enable_negation = False
        # self.categorical_attributes_mask = [] 
        # self.X = []
        self.uX = []
        self.num_of_antecedents_by_attribute = []  # [3, 2, 3]
        self.antecedents_by_attribute = []                 # [(0,1,2),(3,4),(5,6,7)]
        self.attributes_negation_mask = []              # [True, False, True]


    def fit(self,X,categorical_attributes_mask,enable_negation):

        # fuzzify numerical
        X_numerical = X_sample.loc[:,np.invert(categorical_attributes_mask)]

        min_attributes = X_numerical.min()
        max_attributes = X_numerical.max()

        if triangle_format == 'tukey':
            centers = [tukey_centers(x,n_fuzzy_sets) for x in X_numerical.values.T]
        else:
            centers = [normal_centers(mini,maxi,n_fuzzy_sets) for mini,maxi  in zip(min_attributes,max_attributes)]

        fuzzy_sets = np.array([build_fuzzy_sets(x,center,n_fuzzy_sets) for x,center in zip(X_numerical.values.T,centers)]).reshape(X_numerical.shape[0],X_numerical.shape[1] * n_fuzzy_sets)

        # fuzzify categorical
        X_categorical = X_sample.loc[:,categorical_attributes_mask]

        # categories must be provided by the user. In case its empty, it will be calculated automatically (not optimal)
        categories = [np.unique(x) for x in X_categorical.values.T]
        categorical_encoder = OneHotEncoder(categories=categories).fit(X_categorical.values)
        X_categorical_encoded = categorical_encoder.transform(X_categorical.values).toarray()



    def tukey_centers(self, x,n_fuzzy_sets):
        return np.percentile(x, np.linspace(0, 100, n_fuzzy_sets).tolist())

    def normal_centers(self, x_min,x_max,n_fuzzy_sets):
        return np.linspace(x_min, x_max, n_fuzzy_sets)

    def build_fuzzy_sets(self, x,centers,n_fuzzy_sets):

        membership_far_left = self.trapmf(x, [-np.inf, -np.inf, centers[0], centers[1]])
        membership_far_right = self.trapmf(x, [centers[n_fuzzy_sets - 2], centers[n_fuzzy_sets - 1], np.inf, np.inf])
        middle_memberships = np.array([self.trimf(x, centers[i:i + n_fuzzy_sets]) for i in range(n_fuzzy_sets - 2)])
        fuzzy_sets = np.vstack((membership_far_left,middle_memberships,membership_far_right))
        return fuzzy_sets


    def build_membership_functions(self):  
        
        '''
        Build membership functions for attributes
        
        Parameters:
            X: data to fuzzify 
            categorical_attributes_mask: array of booleans indicating which attributes are categorical
            triangle_format: 'normal' or 'tukey'
            n_fuzzy_sets: number of membership functions per attribute. Generally 3,5 or 7
            enable_negation: boolean to enable creation of membership function negations. 
        '''

        list_uX = []
        size_attr = []
        # if self.categorical_attributes_mask:
        for attr in range(self.X.shape[1]):
            if self.categorical_attributes_mask[attr]:
                attribute = pd.DataFrame(self.X[:, [attr]].tolist(), dtype="category")  # print attribute.describe()
                # Talvez trocar esse get_dummier para um encoding que eu tenha a orientação dos dados.
                # Ou ver se tem alguma maneira de fazer isso no get_dummies
                aux = pd.get_dummies(attribute).values
                if aux.shape[1] == 2:  # new IF
                    aux = np.delete(aux, 1, axis=1)
            else:
                attribute = self.X[:, [attr]]
                aux = self.triangle_mb(attribute, self.triangle_format, self.n_fuzzy_sets)

            list_uX.append(aux)
            size_attr.append(aux.shape[1])
        # else:
        #     for attr in range(self.X.shape[1]):
        #         attribute = self.X[:, [attr]]
        #         aux = self.triangle_mb(attribute, self.triangle_format, self.n_fuzzy_sets)
        #         list_uX.append(aux)
        #         size_attr.append(aux.shape[1])

        # list_uX é um tensor: atributos x samples x sets
        self.aux = list_uX
        self.uX = np.hstack(list_uX)
        self.num_of_antecedents_by_attribute = size_attr
        self.antecedents_by_attribute = self.gather_columnspremises_by_attribute(range(self.uX.shape[1]), size_attr)
        self.attributes_negation_mask = self.X.shape[1] * [False]

        if self.enable_negation:
            self.add_negation()
        # return self.uX


    def add_negation(self):

        '''
        Build membership functions  with negation for attributes
        Parameters:
            X: data to fuzzify 
            categorical_attributes_mask: array of booleans indicating which attributes are categorical
            n_fuzzy_sets: number of membership functions per attribute. Generally 3,5 or 7
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



class Premises():

