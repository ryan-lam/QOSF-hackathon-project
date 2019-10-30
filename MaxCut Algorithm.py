#!/usr/bin/env python
# coding: utf-8

# In[48]:


from typing import List, Tuple
import networkx as nx
import numpy as np
from pyquil import get_qc, Program
import csv
from itertools import combinations


# In[87]:


#How to open csv file
with open('../Data/processedIris.csv', 'r') as inputFile: #Opening the csv file
    fileReader = csv.reader(inputFile) #Creating magical file reading 'object' thing
    flower_data = [] #holds the data
    
    for row in fileReader: #goes through each row in data
        flower_data.append(row) #adding row to data holder
    
    inputFile.close(); #closing the original csv file. 

#print(flower_data)

setosa_only = []
for datapoint in flower_data:
    if datapoint[2] == 'Setosa':
        setosa_only.append(datapoint)
#print(setosa_only)

versi_only = []
for datapoint in flower_data:
    if datapoint[2] == 'Versicolor':
        versi_only.append(datapoint)
#print(versi_only)

flower_data = setosa_only[:10] + versi_only[:10]
print(flower_data)
        
tempHeader = flower_data.pop(0)

flower_data_clean = []

for row in flower_data:
    cleanRow = [float(row[0]), float(row[1])]
    flower_data_clean.append(cleanRow)
    
    
#print(flower_data_clean)
    
#print(flower_data[1][:2]) #checking the data loaded  

def distBetweenPoints (point1: list, point2: list) -> float:
    '''
    Inputs: 
    point1 and point2 are lists
    Each is a row in flower_data_clean
    Output: 
    Float distance
    '''
    distX = (point1[0] - point2[0]) ** 2
    distY = (point1[1] - point2[1]) ** 2
    dist = (distX + distY) ** 0.5
    return dist


# In[88]:


#flower_data_clean = [[1,2], [2,3], [4,5], [10,1]]
G = nx.Graph()
datapoints = len(flower_data_clean)

for datapointA in range(len(flower_data_clean)):
    G.add_node(datapointA)

edges = list(combinations(G.nodes, 2))
for edge in edges:
    #print(edge)
    point_a = flower_data_clean[edge[0]]  
    point_b = flower_data_clean[edge[1]]
    #print(distBetweenPoints(point_a, point_b))
    G.add_edge(edge[0], edge[1], weight = distBetweenPoints(point_a, point_b))
               
nx.draw(G)


# In[91]:


#pip install quantum-grove
import numpy as np
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
import pyquil.api as api
qvm_connection = api.QVMConnection()

inst = maxcut_qaoa(G, steps=1, rand_seed=None, connection=None, 
                   samples=None, initial_beta=None, initial_gamma=None, minimizer_kwargs=None, vqe_option=None)
# Sample Run:
# Cutting 0 - 1 - 2 graph!

betas, gammas = inst.get_angles()
probs = inst.probabilities(np.hstack((betas, gammas)))
#for state, prob in zip(inst.states, probs):
#    print(state, prob)

print("Most frequent bitstring from sampling")
most_freq_string, sampling_results = inst.get_string(betas, gammas)
print(most_freq_string)


# In[ ]: