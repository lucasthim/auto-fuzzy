
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .base_fis import BaseFISClassifier

class SENFISClassifier(BaseFISClassifier):

    def __init__(self,triangle_format:str = 'normal',n_fuzzy_sets = 5,enable_negation = False,
                premise_max_size = 2,t_norm = 'prod',criteria_support = 'cardinality',area_threshold = 0.05,
                enable_pcd_premises_base = True, enable_pcd_premises_derived = True,
                enable_similarity_premises_bases = True, enable_similarity_premises_derived = True,
                threshold_similarity = 0.95, association_method = 'MQR', aggregation_method = 'MQR',
                n_classifiers = 10, bagging_fraction = 0.5, feature_fraction = 5, alpha = 0.25, beta = 0.75):

        """
        Implementation of the Selected Ensemble of Automatic Synthesis of Fuzzy Inference Systems (SENFIS) Classifier.
        
        """
        super().__init__(triangle_format = triangle_format, n_fuzzy_sets=n_fuzzy_sets,enable_negation=enable_negation,
        premise_max_size=premise_max_size,t_norm=t_norm,criteria_support=criteria_support,area_threshold=area_threshold,
        enable_pcd_premises_base=enable_pcd_premises_base, enable_pcd_premises_derived=enable_pcd_premises_derived,
        threshold_similarity=threshold_similarity, association_method=association_method, aggregation_method= aggregation_method)

        self.n_classifiers = n_classifiers
        self.bagging_fraction = bagging_fraction
        self.feature_fraction = feature_fraction
        self.alpha = alpha
        self.beta = beta


    def fit(self,X,y = None, categorical_attributes:list = None, verbose = 0):
        # TODO: set feature_fraction default value to sqrt(X.shape[1]) or formula in dissertation (log(features)+1)

        if categorical_attributes is None:
            raise ValueError('categorical_attributes must be set by the user.')
        
        self.categorical_attributes = categorical_attributes
        self.verbose = verbose
        
        self.y_encoder_ = OneHotEncoder(categories = [np.unique(y)], drop='if_binary')
        y_one_hot = self.y_encoder_.fit_transform(y.reshape(-1, 1)).toarray()
        self.num_classes_ = y_one_hot.shape[1]
        self.percentage_of_classes_ = y_one_hot.sum(axis = 0) / y_one_hot.shape[0]
        
        # TODO 0: fuzzify entire X
        self.uX_,self.fuzzy_premises_,self.num_fuzzy_premises_ = self._fuzzify(X)

        print("Done so far!")

        # TODO 1: Create subsets with bag of little bootstrap
        # TODO 2: Get premisses by subset
        # TODO 3: Create autoFIS for each subset (parallel process)
        # TODO 4: Aggregate all rules/classifiers
        # TODO 5: Select classifiers based on n_classifiers and WAD metric
        # TODO 6: Create decision_maker
        pass

    def predict(self,X):
        pass

    

