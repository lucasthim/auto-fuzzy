import numpy as np
from fuzzyfication import Fuzzification
import pandas as pd 



###Carregamento de dados

filedata = pd.read_csv("cluster1.csv")
data = filedata.values

#Definicao de variaveis 

h_prev = 18
lag = 5
lag_notused = []
diff_series = True

bin_values = 12; #Representação da binarização do tempo.

R = 4; #Tamanho maximo de regras.

#####Definicao de funcoes######
detrend_method = ''
bin_method = ''

fuzzy_method = 'mfdef_5triangle'
#Formacao de premissas
form_method = 'form_NonExaustiveSparseBU_v4'
form_param = {R: 4, t_norm:'tnorm_hama' ,card_func:'formativ_supcrop' ,card_min:0.12}


###Pre-processamento de dados
end = data.shape[0]
num_series = data.shape[1]

in_sample = data[0:end-h_prev,:]
out_sample = data[end-h_prev:end,:]

end_insample = in_sample.shape[0]

if diff_series:
    in_sample = in_sample[1:end_insample,:] - in_sample[0:end_insample-1,:]
    end_insample = in_sample.shape[0]

#Fazer detrend method

#Definicao do target
yt = np.zeros([end_insample-lag,num_series],dtype='float')

#Todas as entradas defasadas 
yp = np.zeros([end_insample-lag,num_series], dtype='float')
yp_lagged = np.zeros([end_insample-lag,num_series*lag],dtype='float')


for i in range(num_series):
    yp[:,i] = in_sample[lag:end_insample,i]
    print(i)
    for k in range(1,lag):
        print(k)
        yp_lagged[:,i+k-1] = in_sample[lag-k:end_insample-k,i]




###Fuzzificacao dos dados

Fuzzyfy = Fuzzification('mfdef_5triangle')

mf_params_yp = []

for n in range(num_series):
    
    mX, mf_params = Fuzzyfy.fuzzify(yp[:,n],'')

    mf_params_yp.append(mf_params)


mX_yt = []
mf_params_yt = []

#For output data
for n in range(num_series):
    mX, mf_params = Fuzzyfy.fuzzify(yt[:,n],mf_params[n])

    mX_yt.append(mX_yt)
    mf_params_yt.append(mf_params)

#For lagged data

mX_yp_lagged = []
mf_params_yp_lagged = []

for n in range(num_series):
    for k in range(1,lag):
        mX, mf_params = Fuzzyfy.fuzzify(yp_lagged[:,n+k-1],mf_params[n])

        mX_yp_lagged.append(mX)
        mf_params_yp_lagged.append(mf_params)


#Nao fiz negacao nem binarizacao. Verificar se vou precisar futuramente.

###Inferencia Fuzzy

#Formulacao das regras







