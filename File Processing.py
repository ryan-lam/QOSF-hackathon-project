#!/usr/bin/env python
# coding: utf-8

# In[3]:


from typing import List, Tuple

import networkx as nx
import numpy as np
from pyquil import get_qc, Program
import csv

with open('../Data/iris.csv', 'r') as fileInput: #Reading the data
    fileReader = csv.reader(fileInput)
    flower_data = []
    flower_data.append(next(fileReader)) #handles header before loop
    del(flower_data[0][1])
    del(flower_data[0][2])
    
    for row in fileReader:  #Creating a 2D array (list within list) of data
        filteredRow = [float(row[0]), float(row[2]), row[4]];
        flower_data.append(filteredRow)
        
    fileInput.close()

#Finding necessary variables for normalisation
max0 = 0;
max1 = 0;
min0 = 11110;
min1 = 11110;

tempHeader = flower_data.pop(0) #Temporarily removing header

for row in flower_data:
    if row[0] > max0:
        max0 = row[0];
    elif row[0] < min0:
        min0 = row[0];
     
    if row[1] > max1:
        max1 = row[1];
    elif row[1] < min1:
        min1 = row[1];

for row in flower_data: #normalising
    row[0] = (row[0] - min0) / (max0 - min0);
    row[1] = (row[1] - min1) / (max1 - min1);

flower_data.insert(0, tempHeader); #dding header back

with open('../Data/processedIris.csv', 'w') as outputFile: #outputting new CSV
    fileWriter = csv.writer(outputFile);
    
    for row in flower_data:
        if row[2] == 'Virginica':
            continue;
        fileWriter.writerow(row);
    
    outputFile.close();


# **Use BELOW to Import Processed Data into Projects **

# In[2]:


import csv #Library

#How to open csv file
with open('../Data/processedIris.csv', 'r') as inputFile: #Opening the csv file
    fileReader = csv.reader(inputFile) #Creating magical file reading 'object' thing
    flower_data = [] #holds the data
    
    for row in fileReader: #goes through each row in data
        flower_data.append(row) #adding row to data holder
    
    inputFile.close(); #closing the original csv file. 

print(flower_data[:5]) #checking the data loaded    


# In[ ]: