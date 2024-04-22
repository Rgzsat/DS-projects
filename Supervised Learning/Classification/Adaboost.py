import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score


#data = datasets.load_breast_cancer()
#X, y = data.data, data.target
X, y = datasets.make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

def calc_error(y, y_pred, w_ind):
    '''
    Calculate the error rate of a weak classifier
    y: target variable
    y_pred: predicted variable by weak classifier
    w_ind: individual weights for each observation
        '''
    error= (sum(w_ind * (np.not_equal(y, y_pred)).astype(int)))/sum(w_ind)
    return error

def calc_alpha(error):
    '''
    Through majority vote of the final classifier, it calculate the weight of a weak classifier.
    error: error rate from weak classifier
    '''
    alpha=np.log((1 - error) / error)
    return alpha

def new_weights(w_ind, alpha, y, y_pred):
    ''' 
    Update individual weights w_ind after a boosting iteration
    w_ind: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate new predictions
    '''  
    new_weights= w_ind * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    
    return new_weights

#%%
