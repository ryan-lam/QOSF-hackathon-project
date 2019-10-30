#!/usr/bin/env python
# coding: utf-8

# In[57]:


from typing import List, Tuple

import networkx as nx
import numpy as np
from pyquil import get_qc, Program
import csv


# 

# In[58]:


with open('../Data/iris.csv', 'r') as fileInput:
    fileReader = csv.reader(fileInput)
    flower_data = []
    next(fileReader); #skips header
    
    for row in fileReader:
        flower_data.append(row)
        
    #print(flower_data);
    fileInput.close()


# 

# In[59]:


flower_data_clean = [[0.222222, 0.067797], [0.166667, 0.067797], [0.111111, 0.050847],[0.083333, 0.084746], [0.361111, 0.542373],[0.388889, 0.542373], [0.388889,0.542373], [0.527778,0.559322]]

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


# In[64]:


import networkx as nx

G = nx.Graph()
datapoints = len(flower_data_clean)

for datapointA in range(len(flower_data_clean)):
    G.add_node(datapointA)    
   
largest_weight = 0
edges = list(combinations(G.nodes, 2))
for edge in edges:
    #print(edge)
    point_a = flower_data_clean[edge[0]]  
    point_b = flower_data_clean[edge[1]]
    
    weight = distBetweenPoints(point_a, point_b)
    if weight > largest_weight:
        largest_weight = weight
    #print(distBetweenPoints(point_a, point_b))
    G.add_edge(edge[0], edge[1], weight = weight)
               
nx.draw(G)


# In[66]:


import networkx as nx
import numpy as np
from pyquil.api import get_qc
from pyquil.paulis import PauliTerm, PauliSum
from scipy.optimize import minimize

from grove.pyqaoa.qaoa import QAOA


def maxcut_qaoa(graph, steps=1, rand_seed=None, connection=None, samples=None,
                initial_beta=None, initial_gamma=None, minimizer_kwargs=None,
                vqe_option=None):

    if not isinstance(graph, nx.Graph) and isinstance(graph, list):
        maxcut_graph = nx.Graph()
        for edge in graph:
            maxcut_graph.add_edge(*edge)
        graph = maxcut_graph.copy()
        
    cost_operators = []
    driver_operators = []
    for i, j in graph.edges():
        weight = graph.get_edge_data(i,j)['weight']/largest_weight
        print(weight)
        cost_operators.append(PauliTerm("Z", i, weight)*PauliTerm("Z", j) + PauliTerm("I", 0, -weight))
    for i in graph.nodes():
        driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))

    if connection is None:
        connection = get_qc(f"{len(graph.nodes)}q-qvm")

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}}
    if vqe_option is None:
        vqe_option = {'disp': print, 'return_all': True,
                      'samples': samples}

    qaoa_inst = QAOA(connection, list(graph.nodes()), steps=steps, cost_ham=cost_operators,
                     ref_ham=driver_operators, store_basis=True,
                     rand_seed=rand_seed,
                     init_betas=initial_beta,
                     init_gammas=initial_gamma,
                     minimizer=minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)

    return qaoa_inst


# In[67]:


import numpy as np
import pyquil.api as api
qvm_connection = api.QVMConnection()


# In[71]:


inst = maxcut_qaoa(G, steps=10, rand_seed=None, connection=qvm_connection, samples=None, initial_beta=None, initial_gamma=None, minimizer_kwargs=None, vqe_option=None)


# In[50]:


get_ipython().run_cell_magic('time', '', '\nbetas, gammas = inst.get_angles()\nprobs = inst.probabilities(np.hstack((betas, gammas)))\nfor state, prob in zip(inst.states, probs):\n    print(state, prob)\n\nprint("Most frequent bitstring from sampling")\nmost_freq_string, sampling_results = inst.get_string(betas, gammas)')


# In[51]:


print(most_freq_string)


# In[72]:


import matplotlib.pyplot as plt
plt.scatter(
    x=[a for a,b in flower_data_clean[:4]],
    y=[b for a,b in flower_data_clean[:4]],
    c='r'
)
plt.scatter(
    x=[a for a,b in flower_data_clean[4:]],
    y=[b for a,b in flower_data_clean[4:]],
    c='g'
)


# In[70]:


error = 0

for bit in range(len(most_freq_string)):
    if bit < 4:
        if most_freq_string[bit] == 0:
            continue
        else:
            error+=1
    else:
        if most_freq_string[bit] == 1:
            continue
        else:
            error+=1
errorRate= error/len(most_freq_string)
print("error rate:", errorRate, "%")


# In[ ]: