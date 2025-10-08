
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
# Machine learning algorithms
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

dataset_train = "INSERT YOUR TRAININT PATH HERE"

dataset_valid= "INSERT YOUR VALIDATION PATH HERE"

dataset_test="INSERT YOUR TESTING PATH HERE"

#OPTIONAL, IF YOU HAVE LIMITED DATASETS, IT CONTAINS ALSO A FUNCTION TO SPLIT DATA

train_size = int(len(dataset_train) * 0.70)
val_size= int(len(dataset_train) * 0.20)
test_size= int(len(dataset_train) * 0.10)
scaler = StandardScaler()

def get_val(dataset):
    dataset=(dataset.drop('time', axis=1))
    cap= dataset['mAh']/1000
    max_cap=cap[len(cap)-1]
    dod= (max_cap - cap)/max_cap
    dataset['SOC']= dod
    X= dataset
    X= scaler.fit_transform(X)
    
    ir= (4.1-dataset['V'])/dataset['I']
    yi= ir[0]*dataset['I']+dataset['V']
    y = np.array(yi)
    
    return X,y

X_train, y_train=get_val(dataset_train)[0], get_val(dataset_train)[1]
X_val, y_val=get_val(dataset_valid)[0], get_val(dataset_valid)[1]
X_test, y_test=get_val(dataset_test)[0], get_val(dataset_test)[1]

X_train, y_train=get_val(dataset_train)[0], get_val(dataset_train)[1]
X_val, y_val=get_val(dataset_valid)[0], get_val(dataset_valid)[1]
X_test, y_test=get_val(dataset_test)[0], get_val(dataset_test)[1]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

#PIPELINE, EXAMPLE OF MLP

pipelines = {
    'MLP'   : make_pipeline(StandardScaler(), MLPRegressor(max_iter= 800, random_state=1, #1, or 123
                                                           early_stopping=True))
}

# Check that we have all 5 model families, and that they are all pipelines
for key, value in pipelines.items():
    print( key, type(value) )


# FOR MLP HYPERPARAMETERS

from joblib import dump, load


# MLP hyperparameters
mlp_hyperparameters = {
    'mlpregressor__hidden_layer_sizes': [(200,), (240,), (245,)],#150, 115, 100
    'mlpregressor__activation': ['relu', 'tanh'],
    'mlpregressor__solver': ['adam' ],
        'mlpregressor__alpha': [1],
    'mlpregressor__learning_rate': ['constant','adaptive']
    }


# Create hyperparameters dictionary
hyperparameters = {
    'MLP': mlp_hyperparameters
}

for key in ['MLP']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')

# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=5, n_jobs=-1, return_train_score=True
                         )
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    estimator =  fitted_models[name].best_estimator_
    #dump(estimator, "your-model.joblib")
    joblib.dump(estimator, 'mlp.pkl')


    # Print '{name} has been fitted'
    print(name, 'has been fitted.')

import math
from sklearn.metrics import mean_squared_error
pipe = joblib.load('mlp.pkl')

# Regression Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
for name, model in fitted_models.items():
    print(name, model.best_score_)

for name,model in fitted_models.items():
    #pred_test = fitted_models[name].predict(X_test)
    pred_test = pipe.predict(X_test)
    #print(name)
    #print('R2:', r2_score(y_test, pred_test))
    #print('MAE:', mean_absolute_error(y_test, pred_test))
    #print(('RMSE', math.sqrt(mean_squared_error(y_test, pred_test))))
