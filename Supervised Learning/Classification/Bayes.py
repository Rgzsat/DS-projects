import numpy as np
import math

from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import roc_auc_score

#data = datasets.load_breast_cancer()
#X, y = data.data, data.target
X, y = datasets.make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

n_samples, n_features = X.shape
classes = np.unique(y)
n_classes = len(classes)
ini_mean = np.zeros((n_classes, n_features), dtype=np.float64)
ini_var = np.zeros((n_classes, n_features), dtype=np.float64)
prior_prob =  np.zeros(n_classes, dtype=np.float64)
#%%

def initial_calc(X,y, classes):
    
#Calculation of the mean and variance for each class  
    for i, c in enumerate(classes):
        X_for_class_c = X[y==c]
        ini_mean[i, :] = X_for_class_c.mean(axis=0)
        ini_var[i, :] = X_for_class_c.var(axis=0)
        prior_prob[i] = X_for_class_c.shape[0] / float(n_samples)
    
    return np.concatenate((ini_mean, ini_var), axis=0), prior_prob

f_pos= initial_calc(X, y, classes)

#%%
def calculate_likelihood(class_idx, x):
    mean= f_pos[0][0:2][class_idx]
    #mean = ini_mean[class_idx]
    #var = ini_var[class_idx]
    var= f_pos[0][2:][class_idx]
    num = np.exp(- (x-mean)**2 / (2 * var))
    denom = np.sqrt(2 * np.pi * var)
    likelihood= num/denom
    return likelihood


def classify_sample(x):
     posterior_prob = []
     # calculation of posterior probability for each class
     for i, c in enumerate(classes):
         #fin_prior = np.log(prior_prob[i])
         fin_prior = np.log(f_pos[1][i])
         posterior = np.sum(np.log(calculate_likelihood(i, x)))
         posterior = fin_prior + posterior
         posterior_prob.append(posterior)
     # provides the class with highest posterior probability
     return classes[np.argmax(posterior_prob)]

def naive_predict(X):
    y_pred = [classify_sample(x) for x in X]
    return np.array(y_pred)

#%%
yp =naive_predict(X_test)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=np.array(yp))
#
# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix- Näive Bayes', fontsize=18)
plt.show()

y_true= y_test
print('accuracy is', accuracy_score(y_true, yp))
print('f1 score is', f1_score(y_true,yp, average=None)) 
print('sensitivity is',recall_score(y_true, yp, pos_label=1))
print('specificity is', recall_score(y_true, yp, pos_label=0))
print('precision score is', precision_score(y_true, yp, pos_label= 1))
print('The ROC-AUC score of the model is:', round(roc_auc_score(y_test, yp), 4))

