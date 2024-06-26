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

w_ind = np.ones(len(y_test)) * 1 / len(y_test)  # At m = 0, weights are all the same and equal to 1 / N
alphas = []
boosting_clas = []
training_errors = []
prediction_errors = []
y= y_test
weak_clas= DecisionTreeClassifier(max_depth = 1) 
weak_clas.fit(X_train, y_train)
y_pred = weak_clas.predict(X_test)

#TEST INITIAL FUNCTIONS
error= calc_error(y_test, y_pred, w_ind)
alpha_m= calc_alpha(error)
update_weights= new_weights(w_ind, alpha_m, y_test, y_pred)

#%%
M=150
def ada_boost(X, y, M=M):
        '''
        New model, boosting. Arguments:
        X: independent variables -matrix of features
        y: target variable (to be predicted)
        M: number of boosting rounds.
        '''       
        # Initialize lists and weak classifier
        alphas = []
        boosting_clas = []
        training_errors = []
        w_ind = np.ones(len(y_test)) * 1 / len(y_test)  #At m=0 weights are the same and equal to (1/N)
        weak_clas = DecisionTreeClassifier(max_depth = 1)     # Initialize classification tree
        weak_clas.fit(X, y, sample_weight = w_ind)

        # Iterate over the total of "M" weak classifiers
        for m in range(0, M):
            
            if m == 0:
                w_ind = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                pred_test_i = weak_clas.predict(X_test)
                miss = [int(x) for x in (pred_test_i != y_test)]
                err_m = np.dot(w_ind,miss) / sum(w_ind)
                err_m= calc_error(y_test, pred_test_i, w_ind)
                alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
                w_ind = new_weights(w_ind, alpha_m, y, pred_test_i)

            boosting_clas.append(weak_clas) # Save to list of weak classifiers
            # (b) Calculate error
            error_m = calc_error(y, y_pred, w_ind)
            training_errors.append(error_m)

            # (c) Calculate alpha
            alpha_m = calc_alpha(error_m)
            alphas.append(alpha_m)

        assert len(boosting_clas) == len(alphas)
        
        return alphas, boosting_clas

alphas= ada_boost(X_test, y_test)[0] #take the alphas
final_weak_clas= ada_boost(X_test, y_test)[1] #take the binary weak classifier

def boost_predict(X, M=M):
        '''
        Predict using new model
        X: independent variables
        '''
        # initialize array with predictions of weak classifiers for each observation
        pred_weak = pd.DataFrame(index = range(len(X)), columns = range(M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(M):
            y_pred_m = final_weak_clas[m].predict(X) * alphas[m]
            pred_weak.iloc[:,m] = y_pred_m

        # Estimate final predictions
        y_pred = (1 * np.sign(pred_weak.T.sum())).astype(int)

        #return weak_preds
        return y_pred
    
        
y_boost= boost_predict(X_test)
y_true= y_test

#%%
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn import metrics
import seaborn as sns

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=np.array(y_boost))

# Print the confusion matrix using Matplotlib
cm= metrics.confusion_matrix(y_true, y_boost)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='summer');  
# labels, title and ticks
ax.set_xlabel('Predictions');ax.set_ylabel('Actuals'); 
ax.set_title('Confusion Matrix- AdaBoost'); 

y_true= y_test
print('accuracy is', accuracy_score(y_true, y_boost))
print('f1 score is', f1_score(y_true,y_boost, average=None)) 
print('sensitivity is',recall_score(y_true, y_boost, pos_label=1))
print('specificity is', recall_score(y_true, y_boost, pos_label=0))
print('precision score is', precision_score(y_true, y_boost, pos_label= 1))
print('The ROC-AUC score of the model is:', round(roc_auc_score(y_test, y_boost), 4))

