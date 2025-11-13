import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

import seaborn as sns               #for visualization
import warnings
warnings.filterwarnings('ignore')


bess_df  ="Introduce your dataset"

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

battery_df= df(bess_df)

battery_df.shape
battery_df.info()
battery_df.count()
battery_df.isnull().sum()

battery_df.describe()
sns.boxplot(x = battery_df['OCV'])
plt.show()

#%% Describe the Dataset and removing outliers

battery_df.describe()
sns.boxplot(x = battery_df['OCV'])
plt.show

#%% USING QR TECHNIQUE for outliers

# writing a outlier function for removing outliers in important columns.
def iqr_technique(DFcolumn):
  Q1 = np.percentile(DFcolumn, 25)
  Q3 = np.percentile(DFcolumn, 75)
  IQR = Q3 - Q1
  lower_range = Q1 - (1.5 * IQR)
  upper_range = Q3 + (1.5 * IQR)                        # interquantile range

  return lower_range,upper_range

lower_bound,upper_bound = iqr_technique(battery_df['OCV'])

battery_df = battery_df[(battery_df.OCV>lower_bound) & (battery_df.OCV)]

# so the outliers are removed from price column now check with boxplot and also check shape of new Dataframe!

sns.boxplot(x = battery_df['OCV'])
print(battery_df.shape)

# so here outliers are removed, see the new max price
print(battery_df['OCV'].max())

#%% DATA VISUALIZATION

# Create a figure with a custom size
plt.figure(figsize=(12, 5))

# Set the seaborn theme to darkgrid
sns.set_theme(style='darkgrid')

# Create a histogram of the 'price' column of the battery_df dataframe
# using sns distplot function and specifying the color as red
sns.distplot(battery_df['OCV'],color=('r'))

# Add labels to the x-axis and y-axis
plt.xlabel('OCV [V]', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Add a title to the plot
#plt.title('Distribution of Open Circuit Voltage (OCV)',fontsize=15)

#%%

numerical_features = ['I','V', 'R', 'P', 'mAh', 'wh', 'time', 'SOC', 'OCV']

# Create a boxplot foreach quantitative features

for f in numerical_features:
    sns.boxplot(x=f, data=battery_df )
    plt.show()
    plt.close()

#%%
# Create a histogram for each numerical features
for f in numerical_features:
    sns.histplot(x=f, data= battery_df )
    plt.show()
    plt.close()

#%%

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns,  rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
    

train= battery_df.copy()
plot_corr(train)

#%%
import seaborn as sn

corr = train.select_dtypes(include=['float64', 'int32']).corr()
f, ax = plt.subplots(figsize=(22, 22))
sn.heatmap(corr, vmax=.8, square=True)

#%%
import seaborn as sn

corr = train.select_dtypes(include=['float64', 'int32']).corr()
f, ax = plt.subplots(figsize=(22, 22))
sn.heatmap(corr, vmax=.8, square=True)

#%%
# Correlation between attributes with Voc
corr_list = corr['OCV'].sort_values(axis=0, ascending=False).iloc[1:]
corr_list

#%% Visualize Correlated Attributes

# Scatter plotting the variables most correlated with SalePrice
cols = corr.nlargest(10, 'OCV')['OCV'].index
sn.set()
sn.pairplot(train[cols], size=2.5)
plt.show()

#%%
cf = train.corr()
cf.style.background_gradient(cmap='coolwarm').set_precision(1)
plt.figure(figsize=(12,6)) #9, 8
sn.heatmap(cf, vmax=.8, square=True, annot=True)

#%%
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

target_correlations = train.corr().iloc[-1:, :].transpose().sort_values('OCV', ascending=False).round(2)
target_correlations

# Let's plot the 10 features that have the largest correlation coefficents:
for my_top_10_target_corr in target_correlations.index[1:5]:  
    train.plot(kind="scatter", x=my_top_10_target_corr, y="OCV")

