import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


def f_entr (y):
    ent= 0
    n_labels= len(y)
    if (n_labels<=1):
        ent=0
    probs = []
    for i in set(y): # Set gets a unique set of values. We are iterating over each value
        n_classes = sum(y == i)
        p = n_classes / n_labels # Get the probability of this class label
        probs.append(p)
    
    for j in probs:
        ent= ent-(j*np.log(j)+(1-j)*np.log(1-j))
    return ent
    
def gaussian_generator(center, eig_val, eig_vec, size):
    sigma = np.matmul(np.matmul(eig_vec, eig_val), np.transpose(eig_vec))
    set = np.random.multivariate_normal(center, sigma, size) # observe that here we are using function of the numpy package

    return set

#%%

n_1 = 200 #500  # number of the elements
mu_1 = np.array([2.5, 2])  # center (centroid), #change

# sigma is expected to be positive  semi-definite matrix
w_1 = np.array([[0.6, 0], [0, 0.6]])  # eigenvalues
v_1 = np.array([[0.5, 0.2], [0.2, 0.5]])  # eigenvectors

n_2 = 180 #180  # number of the elements
mu_2 = np.array([1.5, 3])  # center (centroid)

# sigma is expected to be positive  semi-definite matrix
w_2 = np.array([[0.4, 0], [0, 0.4]])  # eigenvalues
v_2 = np.array([[0.6, -0.2], [-0.2, 0.6]])  # eigenvectors

n_3 = 320 #220  # number of the elements
mu_3 = np.array([4.8, -1])  # center (centroid)

# sigma is expected to be positive  semi-definite matrix
w_3 = np.array([[0.2, 0], [0, 0.2]])  # eigenvalues
v_3 = np.array([[-0.8, -0.5], [0.2, -0.8]])  # eigenvectors

set_1 = gaussian_generator(mu_1,w_1, v_1, n_1)
set_2 = gaussian_generator(mu_2,w_2, v_2, n_2)
set_3 = gaussian_generator(mu_3,w_3, v_3, n_3)

# sigma is expected to be positive  semi-definite matrix
w_3 = np.array([[0.2, 0], [0, 0.2]])  # eigenvalues
v_3 = np.array([[-0.8, -0.5], [0.2, -0.8]])  # eigenvectors

#%%

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

y1= np.random.uniform(low=np.min(y), high=np.max(y), size=(400,))
y2= np.random.uniform(low=np.min(y), high=np.max(y), size=(400,))

#y1=np.random.random(400)
#y2=np.random.random(400)
print(f_entr(y))
print(f_entr(y1))
print(f_entr(y2))
print(f_entr(X[:,0]))
print(f_entr(set_1[:,0]))
print(f_entr(set_2[:,0]))
print(f_entr(set_3[:,0]))


#%%

fig_1, ax_1 = plt.subplots()
#a1=ax_1.scatter(set_1[:, 0],  set_1[:, 1], s=10, c='blue')
#a2=ax_1.scatter(set_2[:, 0],  set_2[:, 1], s=10, c='gold')
a3=ax_1.scatter(set_3[:, 0],  set_3[:, 1], s=10, c='green')
a4=ax_1.scatter(X[:, 0],  X[:, 1], s=10, c='pink')
a5= ax_1.scatter(y1,  y2, s=10, c='black')
plt.grid()
plt.ylabel('Second dimension')
plt.xlabel('First dimension')
plt.legend((a3, a4, a5),
           ('E3=6.77', 'E4=7.21', 'E5=6.99'))
plt.show()
