
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame 
#%% DATASETS and loss function

X, y = make_blobs(n_samples=400, centers=2, n_features=4, cluster_std=5, random_state=11)

X = np.c_[np.ones((X.shape[0])), X]
thetas = np.zeros(X.shape[1])

def loss_function(X,y,thetas):
    predictions = np.dot(X,thetas.T)
    
    cost = (1/len(y)) * np.sum((predictions - y) ** 2)
    return cost

loss_f= loss_function(X,y,thetas)
print(mean_squared_error(np.dot(X,thetas.T),y))

#%% 

learn_rate= 0.009
batch_size= 32
def learning_schedule(t, learn_rate, batch_size ):
    return learn_rate/(t+batch_size)

def sgd(X,y,theta,n_epochs):
    hist_cost = [0] * n_epochs
    for epoch in range(n_epochs):
        for i in range(len(y)):
            index_rand = np.random.randint(len(y))
            x_ind = X[index_rand:index_rand+1]
            y_ind = y[index_rand:index_rand+1]

            gradients = 2 * x_ind.T.dot(x_ind.dot(theta) - y_ind)
            eta = learning_schedule(epoch * len(y) + i, learn_rate, batch_size)
            theta = theta - eta * gradients
            hist_cost[epoch] = loss_function(x_ind,y_ind,theta)
    return theta,hist_cost


n_epochs= 200
n_thetas, hist_cost= sgd(X,y,thetas, n_epochs)
print(mean_squared_error(np.dot(X,n_thetas.T),y))

#%% Plot epochs
plt.plot(range(n_epochs),hist_cost)
plt.xlabel('Epochs')
plt.ylabel('Loss function')
plt.show()
#%% Final plots

def graphic(formula, range_x):  
    x = np.array(range_x)  
    y = formula(x)  
    plt.plot(x, y)  
    
def my_formula(x):
    return (-n_thetas[0]-n_thetas[1]*x)/n_thetas[2]

df = DataFrame(dict(x=X[:,2], y=X[:,3], label=y))
colors = {0:'maroon', 1:'forestgreen'}
fig, ax = plt.subplots()
grouped = df.groupby('label')

for i, j in grouped:
    j.plot(ax=ax, kind='scatter', x='x', y='y', label=i, color=colors[i])
graphic(my_formula, range(-20,15))
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()
