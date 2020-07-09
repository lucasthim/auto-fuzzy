import numpy as np
from copy import deepcopy


class Split():

    def __init__(self,mY,prem_terms,num_series):
        self.mY = mY
        self.prem_terms = prem_terms
        self.num_series = num_series

    def run(self,rules):
        return self.FCD(self,rules)





    @staticmethod
    def FCD(self, rules):
        match_degree = np.ndarray(shape=(self.prem_terms.shape[0],self.mY.shape[1],self.mY.shape[2]))
        complete_rules = np.empty(shape=[rules.shape[0], self.num_series],dtype='object')

        for i in range(self.mY.shape[2]):
            
            num_ = np.dot(self.prem_terms,self.mY[:,:,i])
            
            ones = np.ones((num_.shape[0],num_.shape[1]))
            
            
            prem_den = np.sqrt(np.sum(self.prem_terms**2,axis=1))
            
            mY_den = np.sqrt(np.sum(self.mY[:,:,i]**2,axis=0))
            
            
            den1 = ones*prem_den[:,None]
            den2 = (ones.T * mY_den[:,None]).T
            
            den_ = np.multiply(den1,den2)
            match_degree[:,:,i] = np.divide(num_,den_+0.00001)

            best_match = np.argmax(match_degree[:,:,i],axis=1)
            
            for k in range(rules.shape[0]):
                one_rule = deepcopy(rules[k,0])
                one_rule.append((i,best_match[k]))
                complete_rules[k,i] = one_rule


        return complete_rules