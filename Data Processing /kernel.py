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

for i in range(0,len(a1)):
    eu_distance= abs(np.sqrt(a1**2+a2**2))
    distance= np.array(eu_distance)
    if distance[i]< np.quantile(distance, 0.75):
        a3[i]='in'
    else:
        a3[i]='out'

print(distance)

#label encoding
#pd.factorize(a3)
#plt.scatter(a1, a2, c = np.array(pd.factorize(a3)[0])) #2D SEPARATION
#%% Implementing kernel 1

def mapping(a,b):
    x= a**2
    y= b**2
    z= a+b
    f_data= np.column_stack((x,y,z))
    return f_data

f_data= np.array(mapping(a1, a2))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(f_data[:,0], f_data[:,1], f_data[:,2], c=np.array(pd.factorize(a3)[0]))
plt.show()

#%% Implementing kernel 2
from sklearn.datasets import make_moons
colors = ['maroon', 'forestgreen']
features, true_labels = make_moons(n_samples=4000, noise=0.05, random_state=42)
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
#plt.scatter(features[:,0], features[:,1], c=vectorizer(true_labels))

def mapping2(a,b):
    if len(a)>2:
        x=a**3
        y=b**3
        #z= (np.sqrt(6)*(a*b))+(np.sqrt(3)*(a*b))
        z= (np.sqrt(6)*(a**2*b))+(np.sqrt(3)*(a**2*b))
        #z= (x*y)
    else:
            x= a**3
            y= b**3
            z=0
    #f_data2= np.column_stack((x,y,z))
    f_data2= (np.vstack((x, y, z))).T
    return f_data2

f_data2= np.array(mapping2(features[:,0], features[:,1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(f_data2[:,0], f_data2[:,1], f_data2[:,2], c=vectorizer(true_labels))
plt.show()

#%% final plots, together

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf= ax.scatter3D(f_data[:,0], f_data[:,1], f_data[:,2], c=np.array(pd.factorize(a3)[0]))
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(f_data2[:,0], f_data2[:,1], f_data2[:,2], c=vectorizer(true_labels))
plt.show()
