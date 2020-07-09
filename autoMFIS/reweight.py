from utils import *
import numpy as np
from scipy.optimize import minimize



class Reweight():

    def __init__(self,mY,complete_rules, prem_terms):
        self.mY = mY
        self.complete_rules = complete_rules
        self.prem_terms = prem_terms

    def run(self,name,debug=False):
        if name == 'mqr':
            return self.mqr(self,debug) 
        elif name == 'None':
            return self.no_reweigth(self,debug)
        
        else:
            print('Function for reweighting not found.')


    @staticmethod
    def objective_function(x, mY, mX_a):
        global tryout
        #print(x)
        m_diag = np.diag(x)
        #print(x.shape)
        #print(mY.shape)
        #print(mX_a.shape)
        #print(m_diag.shape)
        a = np.sum(np.dot(m_diag,mX_a),axis=0)
        #print(a.shape)
        y = mY - a
        tryout += 1
        if (tryout%100 == 0):
            print('Attempt #{}'.format(tryout))
        return np.sum(np.sqrt(y**2))

    @staticmethod
    def constraint_function(x):
        return np.sum(x) - 1

    @staticmethod
    def mqr(self,debug):
        global tryout

        wd_ = np.zeros(shape=(self.complete_rules.shape[0],self.mY.shape[1],self.mY.shape[2]))

        tryout = 0
        agg_training = np.zeros(shape=self.mY.shape)

        for series in range(self.mY.shape[2]):
            for n_set in range(self.mY.shape[1]):
                index_consequents = np.where(self.mY[:,n_set,series] > 0)

                index_premises = find_rules_by_consequent(self.complete_rules,series,n_set)
                if len(index_consequents) > 0: 
                    if len(index_premises) > 0:
                        filter_prem = self.prem_terms[index_premises,:]

                        activated_prem = filter_prem[:,index_consequents[0]]

                        filtered_consequents = self.mY[index_consequents,n_set,series]
                        tryout = 0
                        if debug:
                            print('---------------------------')
                            print('Shape of activated prem is {}'.format(activated_prem.shape))

                        cons = [{"type": "eq", "fun": self.constraint_function}]

                        bnds = [(0,1) for i in range(filter_prem.shape[0])]

                        res = minimize(self.objective_function,np.ones((activated_prem.shape[0])),args = (filtered_consequents,activated_prem), bounds = bnds, constraints = cons, tol = 1e-2)

                        if debug:
                            print('Shape of initial guess is {}'.format(np.ones((activated_prem.shape[0])).shape))
                            print('Shape of res is {}'.format(res.x.shape))
                            print('Non-zeros weights = {}'.format(np.sum(np.where(res.x>0))))

                        weighted_rules = activated_prem * res.x[:,None]

                        aggr_rules = weighted_rules.max(axis=0)

                        agg_training[index_consequents,n_set,series] = aggr_rules

                        wd_[index_premises,n_set,series] = res.x

        return wd_, agg_training


    @staticmethod
    def no_reweigth(self,debug):
        agg_training = np.zeros(shape=self.mY.shape)
        wd_ = np.ones(shape=(self.complete_rules.shape[0],self.mY.shape[1],self.mY.shape[2]))

        for series in range(self.mY.shape[2]):
            for n_set in range(self.mY.shape[1]):
                index_consequents = np.where(self.mY[:,n_set,series] > 0)

                index_premises = find_rules_by_consequent(self.complete_rules,series,n_set)
                if len(index_consequents) > 0: 
                    if len(index_premises) > 0:
                        filter_prem = self.prem_terms[index_premises,:]

                        activated_prem = filter_prem[:,index_consequents[0]]

                        aggr_rules = activated_prem.max(axis=0)

                        agg_training[index_consequents,n_set,series] = aggr_rules

        return wd_, agg_training