import numpy as np
import matplotlib
from matplotlib.cm import cool
import pandas as pd
#%%

def two_moons(n_data_points, plot):
    rad = 0.6 #radius 
    thk = 0.2 #thickness
    sep = 0.1 #separation
    #calculate the centers
    c1 = np.array([(rad+thk)/2, sep/2])
    c2 =  np.array([-(rad+thk)/2, -sep/2])

#generate data
# We use random radius in the interval [rad, rad+thk]
#  and random angles from 0 to pi[radians].
    r1 = np.random.rand(n_data_points)*thk+rad
    a1 = np.random.rand(n_data_points)*np.pi

    r2 = np.random.rand(n_data_points)*thk+rad
    a2 = np.random.rand(n_data_points)*np.pi+np.pi

# We convert it to cartesian coordinates to plot values:
    p1 = np.array((r1*np.cos(a1), r1*np.sin(a1)))
    p2 = np.array((r2*np.cos(a2), r2*np.sin(a2)))

#previous steps
    x1, y1 = (p1[0] - c1[0], p1[1] - c1[1])
    x2, y2 = (p2[0] - c2[0], p2[1] - c2[1])
    
    set1=np.concatenate((x1,x2), axis=0)
    set2= np.concatenate((y1,y2), axis= 0)
    #np.column_stack((set1,set2))

    moon= (np.vstack((set1, set2))).T
  
    if plot==True:
        
        plt.scatter(x1, y1, marker='.', linewidths=0.1)
        plt.scatter(x2, y2, marker='.', linewidths=0.1)
        plt.show()
    else:
        None
    
    return moon

n_data_points = 2000
import matplotlib.pyplot as plt
print(two_moons(n_data_points, True))

#%%

a1= two_moons(n_data_points, False)[:,0]
a2= two_moons(n_data_points, False)[:,1]
a3= ['general']*len(a1)
count_in = 0 # count of points in inner circle
count_out = 0 # count of points in outer area

