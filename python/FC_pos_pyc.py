## Compare Fastchem, Poseidon and PyChem
#Remove blank spaces in PyChem output # and Pressure.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pych = '/home/pc6/Downloads/pychem_1.dat'
pyc_mix = 'O2S1'
lab = ['PyChem','FastChem']
def reader(pych, pyc_mix,lab):
    read_pyc = pd.read_csv(pych, sep= '\s+', skiprows=0)

    coloms = list(read_pyc.columns)
    #Tval = 2000
    #COval = 0.5
    #metval = 1
    Pval = read_pyc.iloc[:,0]

    
    mix_pyc = read_pyc.iloc[:,coloms.index(pyc_mix)]

    print(Pval,mix_pyc)
    print(coloms)
    plt.loglog(mix_pyc,Pval, label = '{}'.format(lab))

reader(pych, pyc_mix,lab[0])

#FastChem
pyche = '/home/pc6/Downloads/chemistry1.dat'

reader(pyche, pyc_mix,lab[1])
# read_pyc = pd.read_csv(pych, sep= '\s+', skiprows=0)

# coloms = list(read_pyc.columns)
# #Tval = 2000
# #COval = 0.5
# #metval = 1
# Pval = read_pyc.iloc[:,0]

# pyc_mix = 'C1H4'
# mix_pyc = read_pyc.iloc[:,coloms.index(pyc_mix)]

# print(Pval,mix_pyc)
# print(coloms)
# plt.loglog(mix_pyc,Pval, label = 'FastChem')

plt.xlabel('Mixing Ratio of '+pyc_mix)
plt.ylabel('Pressure (bar)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
