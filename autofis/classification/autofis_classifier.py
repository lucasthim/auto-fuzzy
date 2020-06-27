from ..base.fuzzification import Fuzzification
from ..base.formulation import Formulation
from ..base.association import Association
from ..base.aggregation import Aggregation
from ..base.decision import Decision


import numpy as np
from sklearn.preprocessing import OneHotEncoder

class AutoFISClassifier():

    def __init__(self,categorical_attributes = None,triangle_format = 'normal',n_fuzzy_sets = 5,enable_negation = False,
                premise_max_size = 2,t_norm = 'prod',criteria_support = 'cardinalidade_relativa',area_threshold = 0.05,
                enable_pcd_premises_base = True, enable_pcd_premises_derived = True,
                enable_similarity_premises_bases = True, enable_similarity_premises_derived = True,
                threshold_similarity = 0.95, association_method = 'MQR', aggregation_method = 'MQR'):


        '''
        Automatic Synthesis of Fuzzy Inference Systems for Classification
        Proposed by Paredes, J. et al
        DOI: 10.1007/978-3-319-40596-4_41
        Available at: https://www.researchgate.net/publication/303901120_Automatic_Synthesis_of_Fuzzy_Inference_Systems_for_Classification

        This class encapsulates all the steps for the AutoFIS Classifier.


        Parameters
        ----------


        Attributes
        ----------


        '''

        if categorical_attributes is None:
            raise ValueError('categorical_attributes must be set by the user.')
            return;

        self.categorical_attributes = categorical_attributes
        self.triangle_format = triangle_format  # "tukey", "normal"
        self.n_fuzzy_sets = n_fuzzy_sets  # 3, 5, 7
        self.enable_negation = enable_negation


        # Formulation parameters
        self.premise_max_size = premise_max_size
        self.t_norm = t_norm  # "min", "prod"

        # Area filter parameters:
        self.criteria_support = criteria_support  # "cardinalidade_relativa", "frequencia_relativa"
        self.area_threshold = area_threshold
        self.enable_pcd_premises_base = enable_pcd_premises_base
        self.enable_pcd_premises_derived = enable_pcd_premises_derived

        # Overlapping filter parameters:
        self.enable_similarity_premises_bases = enable_similarity_premises_bases
        self.enable_similarity_premises_derived = enable_similarity_premises_derived
        self.threshold_similarity = threshold_similarity

        self.association_method = association_method  # "MQR", "PMQR", "CD", "PCD", "freq_max"
        self.aggregation_method = aggregation_method  # "MQR", "PMQR", "CD", "PCD", "freq_max"



    def fit(self,X,y, verbose = 0):
        self.verbose = verbose
        self.y_encoder = OneHotEncoder(categories = [np.unique(y)], drop='if_binary')
        y_one_hot = self.y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        self.num_classes = y_one_hot.shape[1]
        self.percentage_of_classes = y_one_hot.sum(axis = 0) / y_one_hot.shape[0]
        
        self.uX,self.fuzzy_premises,self.num_fuzzy_premises = self.fuzzify(X)
        self.tree = self.formulate_premises(y_one_hot)
        self.association_rules = self.associate_rules(y_one_hot)
        self.aggregation_rules, self.estimation_classes = self.aggregate_rules(y_one_hot)
        self.decision_maker = self.decision_of_rules()


    def predict(self,X):

        uX_pred,_,_ = self.fuzzifier.transform(X)
        y_pred_one_hot = self.decision_maker.predict(uX_pred,self.t_norm)
        y_pred = self.y_encoder.inverse_transform(y_pred_one_hot)
        return y_pred

    def fuzzify(self,X):
    
        fuzzification_params = {
        'triangle_format' : self.triangle_format,
        'n_fuzzy_sets': self.n_fuzzy_sets,
        'enable_negation': self.enable_negation
        }

        self.fuzzifier = Fuzzification(**fuzzification_params)
        self.fuzzifier.fit(X,is_categorical = self.categorical_attributes)
        # TODO: put fuzzy_premises and num_fuzzy_premises to be calculated INSIDE the formulation step. Not in the fuzzification step.

        if self.verbose:
            print('Done with Fuzzification')
            print('-----------------------')

        return self.fuzzifier.transform(X)

    def formulate_premises(self,y_train_one_hot):

        
        formulation_params = {
        'antecedents_by_attribute': self.fuzzy_premises,
        'num_of_antecedents_by_attribute': self.num_fuzzy_premises,
        'attribute_is_binary': self.fuzzifier.fuzzy_params['is_binary'],
        'ux': self.uX, 
        'target_class': y_train_one_hot,
        'enable_negation': self.enable_negation,
        't_norm': self.t_norm,
        'premise_max_size' : self.premise_max_size,
        'criteria_support' : self.criteria_support, 
        'threshold_support' : self.area_threshold,
        'enable_similarity_premises_bases' : self.enable_similarity_premises_bases,
        'enable_similarity_premises_derived' : self.enable_similarity_premises_derived,
        'threshold_similarity' : self.threshold_similarity,
        'enable_pcd_premises_base' : self.enable_pcd_premises_base,
        'enable_pcd_premises_derived' : self.enable_pcd_premises_derived
        }

        self.formulator = Formulation(**formulation_params)
        tree = self.formulator.generate_premises()
        tree = self.validate_premises(tree)
        
        if self.verbose:
            print('Done with Formulation')
            for i,values in enumerate(tree):
                print('Depth level ' + str(i + 1) + ': ' + str(tree[i][1].shape[1] ))
            print('---------------------')
        return tree;

    def validate_premises(self,tree):

        empty_premises_order = [False if not i[0] else True for i in tree]
        sum_empty_orders = sum(empty_premises_order)
        if sum_empty_orders != len(tree): 
            if sum_empty_orders == 0:
                raise ValueError("Error in Formulation. No premise survived."
                                "It's not possible to continue to the next step."
                                "\nTry to change the parameters of the model.")
            else:
                tree = [i for i in tree if i[0]]
                # arb = [i for i in self.tree if i[0]]
                # tree, arb = arb, tree
        return tree

    def associate_rules(self,y_train_one_hot):
        self.associator = Association(self.tree, y_train_one_hot)
        association_rules = self.associator.build_association_rules(self.association_method)
        self.validate_association_rules(association_rules)
        
        if self.verbose:
            print('Done with Association')
            print('Rules per class:')
            for index,i in enumerate(association_rules):
                print('class ', index, ': ', i[0])
            print('---------------------')

        return association_rules

    def validate_association_rules(self,association_rules):
        status = [0 if not i[0] else 1 for i in association_rules]
        if sum(status) != self.num_classes:
            raise ValueError("Error in Association. Some classes did not get premises. "
                            "It's not possible to continue to the next step."
                            "\nTry to change the parameters of the model.")

    def aggregate_rules(self,y_train_one_hot):

        #TODO: Remove import auxfunc and put everything in one class, in case it is not used anywhere else.
        self.aggregator = Aggregation(self.association_rules)
        aggregation_rules,estimation_classes  = self.aggregator.aggregate_rules(y_train_one_hot, self.aggregation_method)
        self.validate_aggregation_rules(aggregation_rules)
        
        if self.verbose:
            print('Done with Aggregation')
            final_premises_classes = []
            for index,i in enumerate(aggregation_rules):
                print("Premises of Class " + str(index) + ": " + str(i[0]))
                final_premises_classes.append(i[0])
                print("weights: " + str(i[1].T))
            print('-------------------------')

        return aggregation_rules,estimation_classes

    def validate_aggregation_rules(self,aggregation_rules):
        status = [0 if not i[0] else 1 for i in aggregation_rules]
        if sum(status) != self.num_classes:
            raise ValueError("Error in Aggregation. Some classes did not get premises. "
                            "It's not possible to continue to the next step."
                            "\nTry to change the parameters of the model.")

    def decision_of_rules(self):
        decision_maker = Decision(self.aggregation_rules, self.percentage_of_classes)
        if self.verbose:
            print('Done with Decision')
            print('-------------------------')
        return decision_maker