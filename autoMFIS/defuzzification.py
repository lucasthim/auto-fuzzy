import numpy as np
import matplotlib.pyplot as plt
from basicfuzzy import *

class Defuzzification():

    def __init__(self,mf_params,num_series):
        self.mf_params = mf_params
        self.num_series = num_series

    def run(self, name, agg_training, show=False):
        if name == 'cog':
            return self.defuzz_cog(self,agg_training,show) 
        
        else:
            print('Function for Defuzzification not found.')


    @staticmethod
    def defuzz_cog(self,agg_training,show=False):
        y_predict_ = np.zeros((agg_training.shape[0],self.num_series))
        for i in range(self.num_series):

            a = int(self.mf_params[-1,i] - self.mf_params[0,i])
            support_discourse = np.linspace(self.mf_params[0,i],self.mf_params[-1,i],num=a)
            all_values = np.zeros((support_discourse.shape[0],self.mf_params.shape[0]))

            for j in range(self.mf_params.shape[0]):
                if j == 0:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,-1000*abs(self.mf_params[j,i]),-1000*abs(self.mf_params[j,i]),self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1
                    #print(all_values[:,j,i])

                elif j < self.mf_params.shape[0] - 1:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trimf(val,self.mf_params[j-1,i],self.mf_params[j,i],self.mf_params[j+1,i])
                        k += 1

                else:
                    k = 0
                    for val in support_discourse:
                        all_values[k,j] = trapmf(val,self.mf_params[j-1,i],self.mf_params[j,i],1000*abs(self.mf_params[j,i]),1000*abs(self.mf_params[j,i]))
                        k += 1

            for p in range(agg_training.shape[0]):
                p_in = np.ones(shape=all_values.shape) * agg_training[p,:,i]  

                out = np.minimum(all_values,p_in)
                outResponse = np.maximum.reduce(out,axis=1)

                y_predict = sum(np.multiply(support_discourse,outResponse))/(sum(outResponse))

                y_predict_[p,i] = y_predict
                
            if show:
                plt.figure(figsize=(16,9))
                for i in range(all_values.shape[1]):
                    plt.plot(support_discourse,out)
                plt.show()
                plt.close()
            
        return y_predict_
        
