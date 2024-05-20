import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x,y=make_moons(n_samples=550, noise=0.05, random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
plt.scatter(x[:,0],x[:,1],s=25,c=y,cmap=plt.cm.Set1)
plt.show()

#%% Initial helper functions
def nn_inp(inp,out):
    return np.random.randn(inp,out)/np.sqrt(inp)

def create_arch(ini_lay,hid_layer,out_layer,random_seed=0):
    np.random.seed(random_seed)
    layers=x.shape[1],3,1
    arch=list(zip(layers[:-1],layers[1:]))
    nn_weights=[nn_inp(inp,out) for inp,out in arch]
    return nn_weights

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def feed_forward_func(X,nn_weights):
    a=X.copy()
    out_ff=list()
    for w in nn_weights:
        z=np.dot(a,w)
        a=sigmoid_func(z)
        out_ff.append(a)
    return out_ff

def backpropagation_func(h1,h2,nn_weights,y):
    h2_error=y.reshape(-1,1)-h2
    h2_delta=h2_error*sigmoid_derivative(h2)
    h1_error=h2_delta.dot(nn_weights[1].T)
    h1_delta=h1_error*sigmoid_derivative(h1)
    return h2_error,h1_delta,h2_delta
