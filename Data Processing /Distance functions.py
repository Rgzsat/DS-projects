import numpy as np
import matplotlib.pyplot as plt


def minkowski_distance(p1,p2,p):
    di = 0
    for i in range(len(p1)):
        di=di+ abs(p1[i] - p2[i])**p
    distance = di**(1/p)
    return distance


def generate_circle(center, radius, number_of_elements):
    theta = np.linspace(0, 2 * np.pi, number_of_elements)
    circle = np.zeros((number_of_elements, 2))
    for i in range(0, number_of_elements):
        circle[i, 0] = radius * np.cos(theta[i]) + center[0]
        circle[i, 1] = radius * np.sin(theta[i]) + center[1]

    return circle

#%%
number_of_elements = 80
center  = np.array([4, 1])
radius = 5
circle = generate_circle(center, radius, number_of_elements)
minkowski_circle_1= np.zeros((number_of_elements, 2))
minkowski_circle_2= np.zeros((number_of_elements, 2))
minkowski_circle_3= np.zeros((number_of_elements, 2))



theta = np.linspace(0, 2 * np.pi, number_of_elements)
for i in range(0, number_of_elements):
    
    minkowski_circle_1[i, 0] = minkowski_distance(center, circle[i, :],p=90) * np.cos(theta[i]) + center[0]
    minkowski_circle_1[i, 1] = minkowski_distance(center, circle[i, :],p=90) * np.sin(theta[i]) + center[1]
    
    minkowski_circle_2[i, 0] = minkowski_distance(center, circle[i, :],p=2) * np.cos(theta[i]) + center[0]
    minkowski_circle_2[i, 1] = minkowski_distance(center, circle[i, :],p=2) * np.sin(theta[i]) + center[1]
    
    minkowski_circle_3[i, 0] = minkowski_distance(center, circle[i, :],p=1) * np.cos(theta[i]) + center[0]
    minkowski_circle_3[i, 1] = minkowski_distance(center, circle[i, :],p=1) * np.sin(theta[i]) + center[1]


fig_1, axs_1 = plt.subplots()
plt.scatter(center[0], center[1], s=35, c = 'red')
plt.plot(minkowski_circle_1[:, 0], minkowski_circle_1[:, 1], c='black', label='Minkowski, p= 90', linewidth=3)
plt.plot(minkowski_circle_2[:, 0], minkowski_circle_2[:, 1], c='gold', label='Minkowski, p= 2', linewidth=3)
plt.plot(minkowski_circle_3[:, 0], minkowski_circle_3[:, 1], c='blue', label='Minkowski, p= 1', linewidth=3)

axs_1.axis('equal')
plt.legend(loc = 'upper left', framealpha=0.5)
plt.grid()
plt.ylabel('Second dimension')
plt.xlabel('First dimension')
plt.show()

