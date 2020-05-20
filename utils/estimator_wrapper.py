import shap
import pandas as pd
import numpy as np
import copy

class SHAPEstimatorWrapper():

        '''
        SHAP Wrapper for an estimator. 
        
        This class aims to offer the coefficients (coef_) of any estimator as shapley values.

        The coefficients can be used as feature weights and help on feature selection or feature extraction.

        Tip: passing on the SHAP object prior fitting the estimator might lead to faster calculation of coefficients.

        '''

        def __init__(self,EstimatorClass,estimator_params,estimator_type,use_shap, *args, **kwargs):
            self.EstimatorClass = EstimatorClass
            self.estimator_params = estimator_params
            self.estimator = EstimatorClass(**estimator_params)

            self.estimator_type = estimator_type
            self.use_shap = use_shap
            self.shap_coef_ = None
            self.explainer = None
            self.X = None

        def fit(self, *args,**kwargs):
            self.estimator.fit(*args,**kwargs)
            self.X = args[0]

        def predict(self, *args, **kwargs):
            return self.estimator.predict(*args,**kwargs)
        
        def predict_proba(self, *args, **kwargs):
            return self.estimator.predict_proba(*args,**kwargs)
        
        def get_params(self,*args,**kwargs):
            return {
                'EstimatorClass':self.EstimatorClass,
                'estimator_params':self.estimator_params,
                'estimator_type':self.estimator_type,
                'use_shap': self.use_shap
                }

        def _get_tags(self,*args,**kwargs):
            return self.estimator._get_tags();

        @property
        def _estimator_type(self):
            return self.estimator._estimator_type;

        @property
        def coef_(self):
            if self.use_shap:
                return self.get_coef_with_shap(self.X)
            else:
                return self.estimator.coef_

        def get_coef_with_shap(self, X):

            if self.shap_coef_ is None and self.explainer is None:
                n_samples = X.shape[0] if X.shape[0] < 100 else 100;
                
                if self.estimator_type == 'tree':
                    self.explainer = shap.TreeExplainer(self.estimator,X,n_samples = n_samples)[0]
                else:
                    print('Initializing KernelExplainer, this might take a while... \n')
                    self.explainer = shap.KernelExplainer(self.estimator.predict_proba, X, n_samples = n_samples , link="logit")
                # check_additivity = False should be used with caution!!!
                # Check why shap_values from TreeExplainer are coming inside a list
                self.shap_coef_ = np.abs(self.explainer.shap_values(X,check_additivity=False)).sum(axis=0)
            return self.shap_coef_