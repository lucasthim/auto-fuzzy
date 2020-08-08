import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class Fuzzification(BaseEstimator,TransformerMixin):

    '''
    Fuzzification module of a Fuzzy Inference System.

    This transformer class is responsible for fitting and transforming the data into membership functions.
    '''
    
    def __init__(self, n_fuzzy_sets = 3, triangle_format = 'normal',enable_negation = False):

        self.n_fuzzy_sets = n_fuzzy_sets
        self.triangle_format = triangle_format
        self.enable_negation = enable_negation


    def fit(self,X, is_categorical = None, categories = None):

        '''
            Fit fuzzification parameters according to dataset X.
        '''

        X = X if type(X) == pd.DataFrame else pd.DataFrame(X)
        fuzzy_params = pd.DataFrame(columns = ['min','max','centers','is_categorical','is_binary','categories'], index=list(range(X.shape[1])))
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
        fuzzy_params.loc[is_categorical,'is_binary'] = [x.shape[0] <= 2 for x in categories] if categories is not None else False
        fuzzy_params['is_categorical'] = is_categorical

        self.fuzzy_params = fuzzy_params
        

    def transform(self,X):
        
        X = X if type(X) == pd.DataFrame else pd.DataFrame(X)
        uX = []
        premise_ref = 0
        fuzzy_premises = []
        num_fuzzy_premises = []

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
                sets = np.concatenate((sets, 1 - sets), axis=1)

            num_fuzzy_sets = sets.shape[1] if sets.ndim != 1 else 1
            num_fuzzy_premises.append(num_fuzzy_sets)
            # self.fuzzy_params.loc[i,'num_fuzzy_sets'] = num_fuzzy_sets
            fuzzy_premises.append(tuple(range(premise_ref,premise_ref + num_fuzzy_sets)))
            premise_ref += num_fuzzy_sets
            
            uX.append(sets)
        # self.fuzzy_params['fuzzy_sets'] = fuzzy_premises
        uX = np.hstack(uX)
        return uX,fuzzy_premises,num_fuzzy_premises

    def tukey_centers(self, x,n_fuzzy_sets):
        return np.percentile(x, np.linspace(0, 100, n_fuzzy_sets).tolist())

    def normal_centers(self, x_min,x_max,n_fuzzy_sets):
        return np.linspace(x_min, x_max, n_fuzzy_sets)

    def build_fuzzy_sets(self, x,centers,n_fuzzy_sets):

        membership_far_left = self.trapmf(x, [-np.inf, -np.inf, centers[0], centers[1]])
        membership_far_right = self.trapmf(x, [centers[n_fuzzy_sets - 2], centers[n_fuzzy_sets - 1], np.inf, np.inf])
        middle_memberships = np.array([self.trimf(x, centers[i:i + n_fuzzy_sets]) for i in range(n_fuzzy_sets - 2)])
        fuzzy_sets = np.vstack((membership_far_left,middle_memberships,membership_far_right)).T
        return fuzzy_sets


    def triangle_mb(self,y, triangle_format = 'normal', n_fuzzy_sets = 3):

        """
        Build triangular membership functions
        
        Parameters:
        ----------
        y: Attribute list

        triangle_format: 'normal' or 'tukey'
        
        n_fuzzy_sets: number of fuzzy sets

        Returns: 
        ---------

        Array with membership functions
        """

        if triangle_format == 'tukey':
            center = np.percentile(y, np.linspace(0, 100, n_fuzzy_sets).tolist())
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

def main():
    print ('Module 2 <<Fuzzification>>')

if __name__ == '__main__':
    main()