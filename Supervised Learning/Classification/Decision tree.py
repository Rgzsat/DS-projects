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

