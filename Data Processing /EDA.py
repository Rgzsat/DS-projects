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

battery_df= df(airbnb_df)

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

# Create a histogram of the 'price' column of the Airbnb_df dataframe
# using sns distplot function and specifying the color as red
sns.distplot(battery_df['OCV'],color=('r'))

# Add labels to the x-axis and y-axis
plt.xlabel('OCV [V]', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Add a title to the plot
#plt.title('Distribution of Open Circuit Voltage (OCV)',fontsize=15)
