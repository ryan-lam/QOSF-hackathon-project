#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyquil import Program, get_qc, list_quantum_computers
from pyquil.gates import *
from pyquil.api import WavefunctionSimulator
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


p = Program(X(0), CNOT(0,1))
ro = p.declare("ro", "BIT", 2)
p += MEASURE(0,ro[0])
p += MEASURE(1, ro[1])
print(p)

qc = get_qc("4q-qvm")
cp = qc.compile(p)

results = []
for i in range(10):
    state = qc.run(cp)
    results.append(state.tolist())
print(results)



# In[29]:


p = Program(H(0))
results = qc.run_and_measure(p, trials=15)
results


# In[30]:


n_qubits = len(results.keys())
n_trials = len(results[0])


# In[31]:


bitstrings = []
for trial in range(n_trials):
    bitstring = []
    for qubit in range(n_qubits):
        bitstring.append(results[qubit][trial])
    bitstrings.append(bitstring)
        


# In[32]:


bitstrings


# In[ ]: