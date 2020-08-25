import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import accuracy_score

from ..base.fuzzification import Fuzzification
from ..base.formulation import Formulation
from ..base.association import Association
from ..base.aggregation import Aggregation
from ..base.decision import Decision

class BaseFISClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self,triangle_format:str = 'normal',n_fuzzy_sets = 5,enable_negation = False,
                premise_max_size = 2,t_norm = 'prod',criteria_support = 'cardinality',area_threshold = 0.05,
                enable_pcd_premises_base = True, enable_pcd_premises_derived = True,
                enable_similarity_premises_bases = True, enable_similarity_premises_derived = True,
                threshold_similarity = 0.95, association_method = 'MQR', aggregation_method = 'MQR'):
        
        """
        Base class for the Automatic FIS Classifiers.
        This class contains functions that are common to AutoFIS and SENFIS.


        """
        self.triangle_format = triangle_format  # "tukey", "normal"
        self.n_fuzzy_sets = n_fuzzy_sets  # 3, 5, 7
        self.enable_negation = enable_negation

        # Formulation parameters
        self.premise_max_size = premise_max_size
        self.t_norm = t_norm  # "min", "prod"

        # Area filter parameters:
        self.criteria_support = criteria_support  # 'cardinality', 'frequency'
        self.area_threshold = area_threshold
        self.enable_pcd_premises_base = enable_pcd_premises_base
        self.enable_pcd_premises_derived = enable_pcd_premises_derived

        # Overlapping filter parameters:
        self.enable_similarity_premises_bases = enable_similarity_premises_bases
        self.enable_similarity_premises_derived = enable_similarity_premises_derived
        self.threshold_similarity = threshold_similarity

        self.association_method = association_method  # "MQR", "PMQR", "CD", "PCD", "freq_max"
        self.aggregation_method = aggregation_method  # "MQR", "PMQR", "CD", "PCD", "freq_max"



    def fit(self, X,y,**kwargs):
        raise NotImplementedError

    def predict(self,X,**kwargs):
        raise NotImplementedError

    def score(self, X, y = None):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)

    def _fuzzify(self,X):
    
        fuzzification_params = {
        'triangle_format' : self.triangle_format,
        'n_fuzzy_sets': self.n_fuzzy_sets,
        'enable_negation': self.enable_negation
        }

        self.fuzzifier = Fuzzification(**fuzzification_params)
        self.fuzzifier.fit(X,is_categorical = self.categorical_attributes)

        if self.verbose:
            print('Done with Fuzzification')
            print('-----------------------')

        return self.fuzzifier.transform(X)
