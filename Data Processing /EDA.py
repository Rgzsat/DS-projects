import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

import seaborn as sns               #for visualization
import warnings
warnings.filterwarnings('ignore')


airbnb_df  = pd.read_csv(r'C:\Users\47406\Downloads\Coding-initial\1B4-1A.csv')[:-1]

def df(dataset):
    #dataset=(dataset.drop('time', axis=1))
    dataset['time']= dataset['time'].apply(pd.Timedelta).dt.total_seconds().astype(int)
    cap= dataset['mAh']/1000
    max_cap=cap[len(cap)-1]
    dod= (max_cap - cap)/max_cap
    dataset['SOC']= dod
    #X = dataset.drop(['V', 'I'], axis=1)    
    ir= (4.1-dataset['V'])/dataset['I']
    yi= ir[0]*dataset['I']+dataset['V']
    y = np.array(yi)
    dataset['OCV']= y
    
    return dataset

Airbnb_df= df(airbnb_df)

Airbnb_df.shape
Airbnb_df.info()
Airbnb_df.count()
Airbnb_df.isnull().sum()
