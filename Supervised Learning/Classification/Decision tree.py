import pandas as pd
import numpy as np
data= #Insert the data here
#public dataset here: https://github.com/chriswmann/datasets/blob/master/500_Person_Gender_Height_Weight_Index.csv

#%%
#Convert predicted variable to numeric type
data['obese'] = (data.Index >= 4).astype('int').astype('str')
data.drop('Index', axis = 1, inplace = True)

#Initialiation of the loss function

def gini_impurity(v):
  '''
  Given a Pandas Series, it calculates the Gini Impurity of a vector. 
  '''
  vf= np.unique(v, return_counts=True)
  p= vf[1]/v.shape[0]
  #p = yf.value_counts()/yf.shape[0]
  gini = 1-np.sum(p**2)
  return(gini)


def entropy (v):
    vf=  np.unique(v, return_counts=True)
    af= vf[1]/v.shape[0]
    entropy= np.sum(-af*np.log2(af+1e-9))
    return entropy


def variance (v):
    if (len(v)==1):
        return 0
    else:
        vf= np.array((list(map(int,v)))) 
        return (vf.var())

#Test the function
v= data.Gender
gini_impurity(v) 
entropy(v)

variance(data['obese'])
variance(data['Weight'])
variance(data['Height'])
variance(data['Gender']=='Male')

#%%
def information_gain(v, split_c, func=entropy):
  '''
  It returns the Information Gain of a variable given a loss function (entropy or gini).
  v: target variable.
  split_c: split choice.
  func: loss function as input to calculate the parameter "information gain".
  '''
  
  a= np.count_nonzero(split_c)
  b = split_c.shape[0] - a
  
  if(a == 0 or b ==0): 
    inf_gain = 0
  
  else:
    if v.dtypes != 'O':
      inf_gain = variance(v) - (a/(a+b)* variance(v[split_c])) - (b/(a+b)*variance(v[-split_c]))
    else:
      inf_gain = func(v)-a/(a+b)*func(v[split_c])-b/(a+b)*func(v[-split_c])
  
  return inf_gain

#TEST
information_gain(data['obese'], data['Gender'] == 'Male')


def split_max_inf_gain(x, v, func=entropy):
  '''
  Given a predictor and target variable, returns the best split, the error and variable type according to a selected loss function.
  x: independent variable (predictor) as pandas.
  v: predicted variable as pandas Series.
  func: loss function to be used.
  '''

  split_value = []
  inf_gain = [] 

  if x.dtypes!='0':
      numeric_variable= True
  else:
       numeric_variable= False

  # Create options according to variable type
  if numeric_variable:
    options = x.sort_values().unique()[1:]

  # Calculate information gain for all values
  for val in options:
    if numeric_variable:
        mask =   x < val 
    else:
        x.isin(val)
    val_ig = information_gain(v, mask, func)
    # Append results
    inf_gain.append(val_ig)
    split_value.append(val)

  # To check if there are more than 1 results
  if len(inf_gain) == 0:
    return(None,None,None, False)

  else:
  # Get results with highest information gain
    best_ig = max(inf_gain)
    best_ig_index = inf_gain.index(best_ig)
    best_split = split_value[best_ig_index]
    return(best_ig,best_split,numeric_variable, True)

#TEST 
(split_max_inf_gain(data['Weight'], data['obese'],) )
(split_max_inf_gain(data['Height'], data['obese'],) )
(split_max_inf_gain(data['Gender']=='Male', data['obese'],) )
(split_max_inf_gain(data['Gender']=='Female', data['obese'],) )

#%%
#to summarize the split with the predictors, use "apply"
data.drop('obese', axis= 1).apply(split_max_inf_gain, v = data['obese'])

#Get the best split
def calc_best_split(v, data):
  '''
  Select the best split and return the variable, value,variable type, and information gain.
  v: target variable
  data: dataframe to find the best split.
  we define "mf" as a mask that help us to apply the function "get_best_split"
  '''
  mf = data.drop(v, axis= 1).apply(split_max_inf_gain, v = data[v])
  if sum(mf.loc[3,:]) == 0:
    return(None, None, None, None)

  else:
    # Calculate the masks to be splitted in the dataframe
    mf = mf.loc[:,mf.loc[3,:]]

    # Get the results for split with highest information gain
    split_variable = mf.iloc[0].astype(np.float32).idxmax()
    split_value = mf[split_variable][1] 
    split_ig = mf[split_variable][0]
    split_numeric = mf[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)

# TEST
calc_best_split('obese',data)
calc_best_split('Weight',data)
calc_best_split('Gender',data)
calc_best_split('Height',data)

#%%

def execute_split(variable, val, data, is_numeric):
  '''
  Given a data and a split condition, do the split.
  variable: variable with which execute split.
  value: value of the selected variable with which execute the split.
  data: data to be splitted.
  is_numeric: booleanvariable that considers if the variable to be splitted is or not numerical.
  '''
  if is_numeric:
    df_1 = data[data[variable] < val]
    df_2 = data[(data[variable] < val) == False]

  else:
    df_1 = data[data[variable].isin(val)]
    df_2 = data[(data[variable].isin(val)) == False]

  return(df_1,df_2)

def prediction(data, fact_targ):
  '''
  Make a prediction, given an initial dataset.
  data: the target variable, in pandas
  target_factor: Considering if a variable is a factor, input boolean
  '''
  # Make predictions
  if fact_targ:
    predict = data.value_counts().idxmax()
  else:
    predict = data.mean()

  return predict

# TEST
execute_split('Weight', 96, data, True)
execute_split('Height', 149, data, True)
prediction(data['obese'], True)
prediction(data['Weight'], True)
prediction(data['Weight'], False)

#%%

def tree_training(data,v, fact_target, max_depth = None,
               split_min_samples = None, min_inf_gain = 1e-20,
               counter=0):
  '''
  data: Input data
  v: target variable column name
  fact_target: boolean to consider if target variable is factor or numeric.
  max_depth: maximum depth to stop splitting.
  split_min_samples: minimum number of observations to make a split.
  min_inf_gain: minimum ig gain to consider a split to be valid.
  '''

  # Condition to check max_depth
  if max_depth == None:
    depth_cond = True

  else:
    if counter < max_depth:
      depth_cond = True

    else:
      depth_cond = False

  # Condition to check split_min_samples
  if split_min_samples == None:
    sample_cond = True

  else:
    if data.shape[0] > split_min_samples:
      sample_cond = True

    else:
      sample_cond = False

  # Condition to check information gain
  if depth_cond & sample_cond:

    var,val,inf_gain,var_type = calc_best_split(v, data)

    # If information gain condition is met, make split 
    if inf_gain is not None and inf_gain >= min_inf_gain:

      counter= counter+1

      left,right = execute_split(var, val, data,var_type)

      # Initialize sub-tree
      split_type = "<=" if var_type else "in"
      question =   "{} {}  {}".format(var,split_type,val)
      subtree = {question: []}


      # Find answers (recursion)
      posi_answer = tree_training(left,v, fact_target, max_depth,split_min_samples,min_inf_gain, counter)

      neg_answer = tree_training(right,v, fact_target, max_depth,split_min_samples,min_inf_gain, counter)

      if posi_answer == neg_answer:
        subtree = posi_answer

      else:
        subtree[question].append(posi_answer)
        subtree[question].append(neg_answer)

    # If it doesn't match information gain condition, make prediction
    else:
      pred = prediction(data[v],fact_target)
      return pred

   # Drop dataset if doesn't match depth or sample condition
  else:
    pred = prediction(data[v],fact_target)
    return pred

  return subtree

#input variables
max_depth = 5
split_min_samples = 20
min_inf_gain  = 1e-5

#Test
decisions = tree_training(data,'obese',True, max_depth,split_min_samples,min_inf_gain)

#%%

def data_classifier(obs, tree):
  
  '''
  obs: sample of the dataset that will be used in the decision tree.
  tree: output of the previous function, "tree_training"
  output= returns the predicted classifier
  '''
  
  ask_quest = list(tree.keys())[0] 


  if ask_quest.split()[1] == '<=':

    if obs[ask_quest.split()[0]] <= float(ask_quest.split()[2]):
      answer = tree[ask_quest][0]
    else:
      answer = tree[ask_quest][1]

  else:

    if obs[ask_quest.split()[0]] in (ask_quest.split()[2]):
      answer = tree[ask_quest][0]
    else:
      answer = tree[ask_quest][1]


  # If the answer is not a dictionary
  if not isinstance(answer, dict):
    return answer
  else:
    residual_tree = answer
    return data_classifier(obs, answer)

#TEST
data_classifier(data.iloc[3,:], decisions)
data_classifier(data.iloc[6,:], decisions)
data_classifier(data.iloc[180,:], decisions)

#%%

def cross_val(data, n):
    "n= number of selected folds"
    "data= dataset"
    df = data.reindex(np.random.permutation(data.index)) #Randomized data by randomizing the index      
    df = df.reset_index(drop=True) #Randomized data with reseted index

    length = int(len(df)/n) #length of each fold
    folds = []
    for i in range(n-1):
        folds += [data[i*length:(i+1)*length]]
    folds += [data[9*length:len(data)]]
    return folds

#Test
print(cross_val(data, 10)[9])
#%%

def dec_tree(data, y_true, tree):
    data_prediction = []
    num_obser= len(y_true)
    
    for i in range(num_obser):
      obs_pred = data_classifier(data.iloc[i,:], tree)
      data_prediction.append(obs_pred)
    return data_prediction
  
#y_true= data.iloc[1:80,-1]
y_true= (cross_val(data, 10)[9]).iloc[:,-1]
data_f= cross_val(data, 10)[9]
yp=(dec_tree(data_f, y_true, decisions))
print(yp)
#%%

import matplotlib.pyplot as plt
#Accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_true, y_pred=np.array(yp))
#
# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#%%

print('accuracy is', accuracy_score(y_true, yp))
print('f1 score is', f1_score(y_true,yp, average=None)) 
print('sensitivity is', recall_score(y_true, yp, pos_label='1'))
print('specificity is', recall_score(y_true, yp, pos_label='0'))
print('precision score is', precision_score(y_true, yp, pos_label= '1'))
