
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
    
    def __init__(self, n_fuzzy_sets = 3, triangle_format = 'normal',enable_negation = False):

        self.n_fuzzy_sets = n_fuzzy_sets
        self.triangle_format = triangle_format
        self.enable_negation = enable_negation

        self.categorical_attributes_mask = [] 
        self.min_attributes = [] 
        self.max_attributes = [] 
        self.centers = [] 
        self.categories = [] 
        self.categorical_encoder = [] 
        
        self.uX = []
        self.num_of_antecedents_by_attribute = []  # [3, 2, 3]
        self.antecedents_by_attribute = []                 # [(0,1,2),(3,4),(5,6,7)]
        self.attributes_negation_mask = []              # [True, False, True]


    def fit(self,X, is_categorical = None, categories = None):
        
        X = X if type(X) == pd.DataFrame else pd.DataFrame(X)
        fuzzy_params = pd.DataFrame(columns = ['min','max','centers','is_categorical','is_binary','categories','encoder','num_fuzzy_sets','fuzzy_sets'], index=list(range(X.shape[1])))
        is_categorical = is_categorical or X.shape[1] * [False]
        
        X_numerical = X.loc[:,np.invert(is_categorical)]
        if self.triangle_format == 'tukey':
            fuzzy_params.loc[np.invert(is_categorical),'centers'] = [self.tukey_centers(x,self.n_fuzzy_sets) for x in X_numerical.values.T]
        else:
            mins = X_numerical.min()
            maxs = X_numerical.max()
            fuzzy_params.loc[np.invert(is_categorical),'min'] = mins.values
            fuzzy_params.loc[np.invert(is_categorical),'max'] = maxs.values
            fuzzy_params.loc[np.invert(is_categorical),'centers'] = [self.normal_centers(mini,maxi,self.n_fuzzy_sets) for mini,maxi  in zip(mins,maxs)]

        categories = categories or [np.unique(x) for x in X.loc[:,is_categorical].values.T]
        fuzzy_params.loc[is_categorical,'categories'] = categories

        fuzzy_params['is_binary'] = False
        fuzzy_params.loc[is_categorical,'is_binary'] = [x.shape[0] <= 2 for x in categories]
        fuzzy_params['is_categorical'] = is_categorical

        self.fuzzy_params = fuzzy_params
        

    def transform(self,X):
        X = X if type(X) == pd.DataFrame else pd.DataFrame(X)
        
        uX = []
        premise_ref = 0
        # self.fuzzy_params['fuzzy_sets'] = ''
        fuzzy_premises = []
        for i in range(X.shape[1]):
            
            params = self.fuzzy_params.iloc[i]
            x = X.iloc[:,i].values

            if params['is_categorical']:

                sets = OneHotEncoder(categories = [params['categories']],drop='if_binary').fit_transform(x.ravel().reshape(-1,1)).toarray()
                if params['is_binary']:
                    sets = sets.ravel()
            else:
                sets = self.build_fuzzy_sets(x,params['centers'],self.n_fuzzy_sets)

            if self.enable_negation and not params['is_binary']:
                sets = np.concatenate((sets, 1 - sets), axis=0)
            
            num_fuzzy_sets = 1 if sets.ndim == 1 else sets.shape[1]
            self.fuzzy_params.loc[i,'num_fuzzy_sets'] = num_fuzzy_sets
            fuzzy_premises.append(list(range(premise_ref,premise_ref + num_fuzzy_sets)))
            premise_ref += num_fuzzy_sets
            
            uX.append(sets)
        self.fuzzy_params['fuzzy_sets'] = fuzzy_premises
        self.uX
        return uX
        # TODO: Fazer o stack correto ou criar um dataframe de saída....nao sei ainda
        # Os arrays estão "deitados". agrupar todos os arrays deitados e dps transpor
        # return pd.concat(uX)


    # def fit(self,X,categorical_attributes_mask = None, categories = None):

    #     self.categorical_attributes_mask = categorical_attributes_mask or X.shape[1] * [False]
    #     # self.attributes = X.columns if type(X) == pd.DataFrame else list(range(X.shape[1]))

    #     X_numerical = X.loc[:,np.invert(self.categorical_attributes_mask)]
    #     self.min_attributes = X_numerical.min()
    #     self.max_attributes = X_numerical.max()

    #     if self.triangle_format == 'tukey':
    #         self.centers = [self.tukey_centers(x,self.n_fuzzy_sets) for x in X_numerical.values.T]
    #     else:
    #         self.centers = [self.normal_centers(mini,maxi,self.n_fuzzy_sets) for mini,maxi  in zip(self.min_attributes,self.max_attributes)]

    #     X_categorical = X.loc[:,self.categorical_attributes_mask]
    #     self.categories = categories or [np.unique(x) for x in X_categorical.values.T] 
    #     self.categorical_encoder = OneHotEncoder(categories=self.categories,drop='if_binary').fit(X_categorical.values)
    #     self.binary_categories_mask = [x.shape[0] <=2 for x in self.categories]



    # def transform(self,X):

    #     X_numerical = X.loc[:,np.invert(self.categorical_attributes_mask)]
    #     X_numerical_fuzzy = np.array([self.build_fuzzy_sets(x,center,self.n_fuzzy_sets) for x,center in zip(X_numerical.values.T,self.centers)])
    #     X_numerical_fuzzy = X_numerical_fuzzy.reshape(X_numerical.shape[0],X_numerical.shape[1] * self.n_fuzzy_sets)
    #     # test hstack instead of reshape fuzzy_sets

    #     X_categorical = X.loc[:,self.categorical_attributes_mask]
    #     X_categorical_encoded = self.categorical_encoder.transform(X_categorical.values).toarray()
        
    #     fuzzy_sets = np.hstack((X_numerical_fuzzy,X_categorical_encoded))
        
    #     self.premises_numerical = [n_fuzzy_sets for x in range(X_numerical.shape[1])]
    #     self.premises_category = [1 if mask else category.shape[0] for category,mask in zip(self.categories,self.binary_categories_mask)]
        
    #     self.num_fuzzy_sets_by_attribute = self.premises_numerical + self.premises_category
    #     self.fuzzy_sets_by_attribute = self.group_fuzzy_sets_by_attribute(range(fuzzy_sets.shape[1]),self.num_fuzzy_sets_by_attribute)
        
    #     if self.enable_negation:
    #         self.add_negation_to_fuzzy_sets(X,fuzzy_sets)
    #     return fuzzy_sets


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


    # def add_negation_to_fuzzy_sets(self,X,fuzzy_sets):
    #     # Aqui é pra inverter todo o fuzzy_sets e filtrar apenas as colunas de interesse
    #     # Also, indicar quais os antecedentes de cada atributo
    #     attributes_to_negate = [not(categorical) or (categorical and np.unique(x).shape[0] > 2) for x,categorical in zip(X.values.T,self.categorical_attributes_mask)]
        
 
    #     sets_to_negate = pd.Series(self.fuzzy_sets_by_attribute)[attributes_to_negate] 
    #     fuzzy_sets_to_negate = [i for sub in list(sets_to_negate) for i in sub] 
    #     negated_fuzzy_sets = 1 - fuzzy_sets

    #     new_fuzzy_sets = np.concatenate((fuzzy_sets, negated_fuzzy_sets[:, fuzzy_sets_to_negate]), axis=1)
    #     return new_fuzzy_sets
    #     # self.antecedents_by_attribute = total_premises[:]
    #     # self.num_of_antecedents_by_attribute = [len(i) for i in total_premises]

    # def group_fuzzy_sets_by_attribute(self, attributes, n_fuzzy_sets_per_attribute):
        
    #     """
    #     Gather columnspremises by each attribute
        
    #     Parameters:
            
    #         attributes: Array with number of premises [0,1,2,3...,7]
            
    #         n_fuzzy_sets_per_attribute: Array with membership functions per attribute [3, 2, 3]
            
    #         n_fuzzy_sets: number of fuzzy sets
            
    #     Return: array with each premise grouped in a tuple [(0,1,2), (3,4), (5,6,7)]
    #     """

    #     new_attributes = []
    #     ref = 0
    #     for i in n_fuzzy_sets_per_attribute:
    #         new_attributes.append(tuple(attributes[ref:ref+i]))
    #         ref += i
    #     return new_attributes

    # def build_membership_functions(self):  
        
    #     '''
    #     Build membership functions for attributes
        
    #     Parameters:
    #         X: data to fuzzify 
    #         categorical_attributes_mask: array of booleans indicating which attributes are categorical
    #         self.triangle_format: 'normal' or 'tukey'
    #         n_fuzzy_sets: number of membership functions per attribute. Generally 3,5 or 7
    #         enable_negation: boolean to enable creation of membership function negations. 
    #     '''

    #     list_uX = []
    #     size_attr = []
    #     for attr in range(self.X.shape[1]):
    #         if self.categorical_attributes_mask[attr]:
    #             attribute = pd.DataFrame(self.X[:, [attr]].tolist(), dtype="category")  # print attribute.describe()
    #             aux = pd.get_dummies(attribute).values
    #             if aux.shape[1] == 2:  # new IF
    #                 aux = np.delete(aux, 1, axis=1)
    #         else:
    #             attribute = self.X[:, [attr]]
    #             aux = self.triangle_mb(attribute, self.self.triangle_format, self.n_fuzzy_sets)

    #         list_uX.append(aux)
    #         size_attr.append(aux.shape[1])

    #     # list_uX é um tensor: atributos x samples x sets
    #     self.aux = list_uX
    #     self.uX = np.hstack(list_uX)
    #     self.num_of_antecedents_by_attribute = size_attr
    #     self.antecedents_by_attribute = self.gather_columnspremises_by_attribute(range(self.uX.shape[1]), size_attr)
    #     self.attributes_negation_mask = self.X.shape[1] * [False]

    #     if self.enable_negation:
    #         self.add_negation()
    #     # return self.uX


    # def add_negation(self):

    #     '''
    #     Build membership functions  with negation for attributes
    #     Parameters:
    #         X: data to fuzzify 
    #         categorical_attributes_mask: array of booleans indicating which attributes are categorical
    #         n_fuzzy_sets: number of membership functions per attribute. Generally 3,5 or 7
    #         enable_negation: boolean to enable creation of membership function negations. 
    #     '''
        
    #     num_attributes = len(self.categorical_attributes_mask)
    #     num_col = sum(self.num_of_antecedents_by_attribute) 

    #     attr_with_more_than_2_membership_functions = [i > 2 for i in self.num_of_antecedents_by_attribute] 
    #     # attr_with_more_than_2_membership_functions = [False, True, False, True, True]
    #     # categorical_attributes_mask = [True, False, True, False,True]
        
    #     index_premises_negation = [(attr_with_more_than_2_membership_functions[i] + 1 - self.categorical_attributes_mask[i]) != 0 for i in range(len(self.categorical_attributes_mask))] 
    #     self.attributes_negation_mask = index_premises_negation
    #     # Atributos com mais de 2 membership functions OU atributos que NÃO são categoricos
    #     # [(0 + 1 - 1) != 0 ] -> [False]
    #     # [(1 + 1 - 0) != 0 ] -> [True]
    #     # [(0 + 1 - 1) != 0 ] -> [False]
    #     # [(1 + 1 - 0) != 0 ] -> [True]
    #     # [(1 + 1 - 1) != 0 ] -> [True]
        
    #     attrib_survivors_negation = list(compress(range(num_attributes), index_premises_negation)) 
    #     # [1,3,4]

    #     # num_of_antecedents_by_attribute = [2,3,2,3,3]
    #     # list(pd.Series(self.num_of_antecedents_by_attribute)[attrib_survivors_negation]) -> [3,3,3]
    #     premises_attrib_neg = self.gather_columnspremises_by_attribute(range(num_col, 2*num_col),list(pd.Series(self.num_of_antecedents_by_attribute)[attrib_survivors_negation]))
    #     # lista de tuplas que combinam os fuzzy sets com os atributos de X
    #     # [(15,16,17),(20,21,22),(23,24,25)]

    #     premises_survivors_negation = list(compress(premises_attrib_neg, list(pd.Series(self.num_of_antecedents_by_attribute)[attrib_survivors_negation])))
    #     # [(15,16,17),(20,21,22),(23,24,25)]

    #     total_premises = []
    #     for i in range(num_attributes):
    #         #antecedents_by_attribute = [(0,1),(2,3,4),(5,6),(7,8,9),(10,11,12)]
    #         prem_attr_i = self.antecedents_by_attribute[i]
    #         if i in attrib_survivors_negation:
    #             aux_index = attrib_survivors_negation.index(i)
    #             prem_attr_i += premises_survivors_negation[aux_index]
    #         total_premises.append(prem_attr_i)
        
    #     prem_surv = pd.Series(self.antecedents_by_attribute)[attrib_survivors_negation] 
    #     ind_neg = [i for sub in list(prem_surv) for i in sub] 
        
    #     self.uX = np.concatenate((self.uX, 1. - self.uX[:, ind_neg]), axis=1)
    #     self.antecedents_by_attribute = total_premises[:]
    #     self.num_of_antecedents_by_attribute = [len(i) for i in total_premises]
    #     # return self.uX

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


    # def gather_columnspremises_by_attribute(self, premises_list, sizes_attributes):
        
    #     """
    #     Gather columnspremises by each attribute
        
    #     Parameters:
            
    #         premises_list: Array with number of premises [0,1,2,3...,7]
            
    #         sizes_attributes: Array with membership functions per attribute [3, 2, 3]
            
    #         n_fuzzy_sets: number of fuzzy sets
            
    #     Return: array with each premise grouped in a tuple [(0,1,2), (3,4), (5,6,7)]
    #     """

    #     new_premises_list = []
    #     ref = 0
    #     for i in sizes_attributes:
    #         new_premises_list.append(tuple(premises_list[ref:ref+i]))
    #         ref += i
    #     return new_premises_list


# class Premises():

# pass