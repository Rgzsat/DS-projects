import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob

path = # Use your path here
#all_files = glob.glob(os.path.join(path , "/*.csv"))
files = Path(path).glob('*.csv')  # .rglob to get subdirectories
from sklearn.preprocessing import MinMaxScaler

datasets = list()
for f in files:
    data = pd.read_csv(f, index_col=6)
    # .stem is method for pathlib objects to get the filename w/o the extension
    #data['file'] = f.stem
    datasets.append(data)

scaler = "Introduce your path here"

def df(dataset):   
    df1= dataset.iloc[1: , :]
    cap= df1['mAh']/1000
    max_cap=cap[len(cap)-1]
    dod= (max_cap - cap)/max_cap
    df2= df1.assign(SOC= dod)
    X= scaler.fit_transform(df2)
    
    ir=(dataset['V'][0]-df2['V'])/df2['I'] 
    yi= ir[0]*df2['I']+df2['V']
    y = np.array(yi)
    y = np.reshape(y, (-1, 1))
    y= scaler.fit_transform(y)
    
    return X, y, df2

x_test= list()
X_test=list()
y_test= list()
soc= list()

for i in range(0,len(datasets)):
    xtest= df(datasets[i])[0]
    x_test.append(xtest)
    Xtest= x_test[i].reshape((x_test[i].shape[0], 1, x_test[i].shape[1]))
    X_test.append(Xtest)
    
    ytest= df(datasets[i])[1]
    y_test.append(ytest)
    
    soc_ini=(df(datasets[i])[2].SOC)*100
    soc.append(soc_ini)

#%%
import keras
from sklearn.metrics import r2_score
#from tensorflow.keras.models import load_model

bigru= "Introduce your path here"
bilstm= "Introduce your path here"

yf= list()

bigru_predict= list()
bigru_evaluate= list()
bigru_r2= list()

bilstm_predict= list()
bilstm_evaluate= list()
bilstm_r2= list()


for j in range(0, len(datasets)):
    yf.append(scaler.inverse_transform(y_test[j]))
    
    ocv_bigru= bigru.predict(X_test[j])
    ocv_bigru= scaler.inverse_transform(ocv_bigru)
    bigru_predict.append(ocv_bigru)
    metrics_bigru= bigru.evaluate(x=X_test[j],y=y_test[j])
    bigru_evaluate.append(metrics_bigru)
    bigru_r2.append(r2_score(yf[j], bigru_predict[j]))
    
    ocv_bilstm= bilstm.predict(X_test[j])
    ocv_bilstm= scaler.inverse_transform(ocv_bilstm)
    bilstm_predict.append(ocv_bilstm)
    metrics_bilstm= bilstm.evaluate(x=X_test[j],y=y_test[j])
    bilstm_evaluate.append(metrics_bilstm)
    bilstm_r2.append(r2_score(yf[j], bilstm_predict[j]))
    
#https://pandas.pydata.org/docs/user_guide/merging.html
col= ['loss', 'mse', 'mae']
r2= ['r2']

final_bilstm= pd.concat([pd.DataFrame(bilstm_evaluate, columns= col)
                         ,pd.DataFrame(bilstm_r2, columns= r2) ], axis=1, join="inner")

final_bigru= pd.concat([pd.DataFrame(bigru_evaluate, columns= col)
                         ,pd.DataFrame(bigru_r2, columns= r2) ], axis=1, join="inner")

print(final_bilstm.mean(axis=0))


