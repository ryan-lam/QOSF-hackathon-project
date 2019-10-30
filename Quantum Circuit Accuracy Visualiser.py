#!/usr/bin/env python
# coding: utf-8

# In[33]:


from pyquil import Program, get_qc, list_quantum_computers
from pyquil.gates import *
from pyquil.api import WavefunctionSimulator
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# In[45]:


p = Program()    # set up a cirquit
ro = p.declare('ro', 'BIT', 2)     # declare memory space ro
p += X(0)     # apply X gate on qubit 0 (rotation around x axis)
p += Y(0)     # apply Y gate on qubit 0
p += CNOT(0, 1)     # adding CNOT between  qubit 0 and 1
p += MEASURE(1, ro[1])     # measure
p += MEASURE(0, ro[0])     # measure

#print(p)

#wfn = WavefunctionSimulator().wavefunction(p)
#print(wfn)
#print(list_quantum_computers())   #lists all available quantum computers and simulators
#qc = get_qc('9q-square-qvm')
qc = get_qc('9q-square-noisy-qvm')

#print(qc)
##cp = qc.compile(p)

results = []
for i in range(50):
    state = qc.run(cp)
    results.append(state.tolist())
print(results)

unique =[]
for x in results:
    if x not in unique:
        unique.append(x)
print(unique)
    
performance = []
for i in range(len(unique)):
    print(str(unique[i]) + " = " + str(results.count(unique[i])) + " (" +str(((results.count(unique[i]))/len(results))*100) + "%)")
    performance.append((((results.count(unique[i]))/len(results))*100))
        
objects = []

for x in range(len(unique)):
    objects.append(x)
    
y_pos = np.arange(len(unique))

plt.bar(y_pos, performance, align='center', alpha=0.1)
plt.xticks(y_pos, unique)
plt.ylabel('%')
plt.title('x')

plt.show()
    
    


# In[ ]:





# In[ ]: