#!/usr/bin/env python
# coding: utf-8

# In[6]:


#FUN LIBRARIES AND STUFF FOR FREE, KIDS

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy



#Setting up dataset and graphs

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data = pd.read_csv('../Data/processedIris.csv')
#data.head()

first4 = data[:4]
last4 = data[len(data)-6:len(data)-2]
frames = [first4, last4]
new_data = pd.concat(frames)
new_data.values.tolist()
data = new_data

listified = data['variety'].tolist()
print(data)

for category in range(len(listified)):
    if listified[category] == 'Setosa':
        listified[category] = 1;
    else:
        listified[category] = 0;


# In[30]:




# Getting the values and plotting it

ft1 = data['sepal.length'].values
ft2 = data['petal.length'].values
X = np.array(list(zip(ft1, ft2)))
plt.scatter(ft1, ft2, c='black', s=7)


# In[31]:


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)




#WHEN YOU DON'T KNOW WHAT YOU WANT IN LIFE SO YOU JUST SET THINGS TO RANDOM

# Number of clusters
k = 2
# X coordinates of random centroids
Centroid_x = np.random.uniform(0, np.max(X), size=k)
# Y coordinates of random centroids
Centroid_y = np.random.uniform(0, np.max(X), size=k)
Centroid = np.array(list(zip(Centroid_x, Centroid_y)), dtype=np.float32)
print(Centroid)


# In[32]:


# Plotting along with the Centroids
plt.scatter(ft1, ft2, c='#050505', s=7)
plt.scatter(Centroid_x, Centroid_y, marker='*', s=200, c='g')


# In[33]:


get_ipython().run_cell_magic('time', '', '# To store the value of centroids when it updates\nCentroid_old = np.zeros(Centroid.shape)\n# Cluster Lables(0, 1, 2)\nclusters = np.zeros(len(X))\n# Error func. - Distance between new centroids and old centroids\nerror = dist(Centroid, Centroid_old, None)\n\n\n\n\n\n# Loop will run till the error becomes zero\nwhile error != 0:\n    # Assigning each value to its closest cluster\n    for point in range(len(X)): #Each point (pair of two features)\n        distances = dist(X[point], Centroid) #Dist between centroids and point\n        cluster = np.argmin(distances) #Smallest dist (which point cluster belongs to)\n        clusters[point] = cluster\n        \n    # Storing the old centroid values\n    Centroid_old = deepcopy(Centroid)\n    # Finding the new centroids by taking the average value\n    for i in range(k):\n        points = [X[point] for point in range(len(X)) if clusters[point] == i]\n        Centroid[i] = np.mean(points, axis=0)\n    error = dist(Centroid, Centroid_old, None)\n    \nverify = clusters.tolist();\n\n\n\n\nnumErrors = 0;\nfor i in range(len(verify)):\n    if int(verify[i]) == listified[i]:\n        continue;\n    else:\n        numErrors += 1;\n\nprint( "Error Rate: " + str(numErrors / len(verify)));\n\n\n\n\n\n#Plotting the new centroids\ncolors = [\'r\', \'g\']\nfig, ax = plt.subplots()\nfor i in range(k):\n        points = np.array([X[point] for point in range(len(X)) if clusters[point] == i])\n        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])\nax.scatter(Centroid[:, 0], Centroid[:, 1], marker=\'*\', s=200, c=\'#050505\')')


# In[ ]: