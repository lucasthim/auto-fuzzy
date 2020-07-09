#Fuzzification methods. Depends on Basic Fuzzy functions.

from basicfuzzy import trimf, trapmf
import numpy as np
from numpy import array, quantile
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Fuzzification:
    
    def __init__(self,name):
        self.name = name
        #print(self.name)
    
    def fuzzify(self,values,mf_params,num_groups=5):
        if self.name is 'mfdef_triangle':
            mX, mf_params = self.mfdef_triangle(self,values,mf_params,num_groups = num_groups)
            return mX, mf_params

        elif self.name is 'mfdef_tukkey':
            mX, mf_params = self.mfdef_tukkey(self,values,mf_params,num_groups = num_groups)
            return mX, mf_params

        elif self.name is 'mfdef_cluster':
            mX, mf_params = self.mfdef_cluster(self,values,mf_params,cluster=num_groups)
            return mX, mf_params

        else:
            print("Something went wrong. Please verify Fuzzyfication")

    def draw_fuzzy(self,mf_params,name):
            i = 0

            values = np.linspace(mf_params[0],mf_params[-1],1000)
            print(mf_params.shape)
            mX = np.zeros([len(values),mf_params.shape[0]], dtype='float')

            for val in values:
                
                mX[i,0] = trapmf(val,-1000*abs(mf_params[0]),-1000*abs(mf_params[0]),mf_params[0],mf_params[1])
                for j in range(1,mf_params.shape[0]-1):
                    mX[i,j] = trimf(val,mf_params[j-1],mf_params[j],mf_params[j+1])
                
                mX[i,j+1] = trapmf(val,mf_params[j],mf_params[j+1],1000*abs(mf_params[j+1]),1000*abs(mf_params[j+1]))
                i += 1

            plt.figure()
            f,ax=plt.subplots(figsize=(16,5))
            for i in range(mX.shape[1]):
                ax.plot(values,mX[:,i])
                ax.set(xlabel='Values',ylabel=r"$\mu(x)$",title='Fuzzy sets using {}'.format(name))
            





    @staticmethod
    def mfdef_triangle(self,values,mf_params, num_groups=5):

        if mf_params.size == 0:
            
            mf_params = [h for h in np.linspace(np.min(values),np.max(values),num_groups)]

            mX = np.zeros([values.shape[0],num_groups], dtype='float')
            i = 0
            for val in values:
                
                mX[i,0] = trapmf(val,-1000*abs(mf_params[0]),-1000*abs(mf_params[0]),mf_params[0],mf_params[1])
                for j in range(1,num_groups-1):
                    mX[i,j] = trimf(val,mf_params[j-1],mf_params[j],mf_params[j+1])
                
                mX[i,j+1] = trapmf(val,mf_params[j],mf_params[j+1],1000*abs(mf_params[j+1]),1000*abs(mf_params[j+1]))

                i += 1

            
            
           
            return mX, np.array(mf_params)

        else:
            
            mX = np.zeros((values.shape[0],num_groups), dtype='float')
            i = 0
            for val in values:
                
                mX[i,0] = trapmf(val,-1000*abs(mf_params[0]),-1000*abs(mf_params[0]),mf_params[0],mf_params[1])
                for j in range(1,num_groups-1):
                    mX[i,j] = trimf(val,mf_params[j-1],mf_params[j],mf_params[j+1])
                
                mX[i,j+1] = trapmf(val,mf_params[j],mf_params[j+1],1000*abs(mf_params[j+1]),1000*abs(mf_params[j+1]))

                i += 1

            
            
            return mX, mf_params

    @staticmethod
    def mfdef_tukkey(self,values,mf_params,num_groups = 5):

        if mf_params.size == 0:

            a = quantile(values,0)
            e = quantile(values,1)
            c = quantile(values,0.5)
            d = quantile(values,0.75)
            b = quantile(values,0.25)

            mX = np.zeros([len(values),5], dtype='float')
            i = 0
            for val in values:
                
                mX[i,0] = trapmf(val,-1000*abs(a),-1000*abs(a),a,b)
                mX[i,1] = trimf(val,a,b,c)
                mX[i,2] = trimf(val,b,c,d)
                mX[i,3] = trimf(val,c,d,e)
                mX[i,4] = trapmf(val,d,e,1000*abs(e),1000*abs(e))

                i += 1

            mf_params = array([a,b,c,d,e])

            return mX, mf_params

        else:
            a,b,c,d,e = mf_params

            mX = np.zeros([len(values),5], dtype='float')
            i = 0
            for val in values:
                mX[i,0] = trapmf(val,-1000*abs(a),-1000*abs(a),a,b)
                mX[i,1] = trimf(val,a,b,c)
                mX[i,2] = trimf(val,b,c,d)
                mX[i,3] = trimf(val,c,d,e)
                mX[i,4] = trapmf(val,d,e,1000*abs(e),1000*abs(e))

                i += 1

            
            
            return mX, mf_params

    @staticmethod
    def mfdef_cluster(self,values,mf_params,cluster='5'):

        v = np.array(values).reshape(-1,1)

        if mf_params.size == 0:
            
            kmeans = KMeans(n_clusters=cluster, random_state=0).fit(v)
            centers = kmeans.cluster_centers_
            centers.sort(axis=0)
            mf_params = centers 
            mX = np.zeros((values.shape[0],cluster), dtype='float')
            i = 0
            for val in values:
                
                mX[i,0] = trapmf(val,-1000*abs(mf_params[0]),-1000*abs(mf_params[0]),mf_params[0],mf_params[1])
                for j in range(1,cluster-1):
                    mX[i,j] = trimf(val,mf_params[j-1],mf_params[j],mf_params[j+1])
                
                mX[i,j+1] = trapmf(val,mf_params[j],mf_params[j+1],1000*abs(mf_params[j+1]),1000*abs(mf_params[j+1]))

                i += 1

            
            
            
            #mf_params = array([a,b,c,d,e])
            return mX, mf_params

        else:
            
            mX = np.zeros((values.shape[0],cluster), dtype='float')
            i = 0
            for val in values:
                
                mX[i,0] = trapmf(val,-1000*abs(mf_params[0]),-1000*abs(mf_params[0]),mf_params[0],mf_params[1])
                for j in range(1,cluster-1):
                    mX[i,j] = trimf(val,mf_params[j-1],mf_params[j],mf_params[j+1])
                
                mX[i,j+1] = trapmf(val,mf_params[j],mf_params[j+1],1000*abs(mf_params[j+1]),1000*abs(mf_params[j+1]))

                i += 1

            
            
            return mX, mf_params
        
