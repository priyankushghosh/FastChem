#Compare with Poseidon

import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
#%matplotlib qt

rcParams['axes.labelsize'] = 18 #15
rcParams['axes.titlesize'] = 18
rcParams['xtick.labelsize'] = 18# 15
rcParams['ytick.labelsize'] = 18 #15
rcParams['legend.fontsize'] = 16

#Poseidon tuple: (Met, C_O, T, P)
pos_tuple = ['T','P','C_O','Met'] #['Met','C_O','T','P'] #Change accordingly with pos_arr 

fixed_met = 1 #1e-1
fixed_co = 0.5 #2e-1
fixed_T = 1500 #3e2
which_molc = 'H2S1'
pos_molc = 'H2S'
mix_name = 'MR' #Name of the hdf5 dataset (MR or log(X))
file_fc = '/home/pc6/FastChem/output_pyfc/trials_nexotrans/full_grid_tiv_exact_poseidon_asp2009.h5'
file_pos = '/home/pc6/Downloads/fastchem_database.hdf5'

def h5read(filen,which_molc, fixed_T,fixed_co,fixed_met, pos_tuple, mix_name, posq):
    mixer =[]
    with h5py.File(filen,'r') as pos:
        s = list(pos.keys())
        print(s)
        grp = pos[which_molc]
        dset = list(grp.keys())
        print(dset)
        ds = grp[mix_name][:]
        print(ds, len(ds), min(ds))
        # data = list(pos[s])
        # print(data)
        
        #Find length of grids
        pars = pos['Info']
        pars_dset = list(pars.keys())
        t = pars['T grid'][:]
        t_len = len(t)
        p = pars['P grid'][:]
        p_len = len(p)
        co = pars['C']['O grid'][:]
        co_len = len(co)
        mett = pars['M']['H grid'][:]
        mett_len = len(mett)
        '''pos_arr = [t_len, p_len, co_len, mett_len]
        print(p_len,t_len,co_len,mett_len)'''
        which_is_variable = pos_tuple.index('P')
        
        #If four variables are (a,b,c,d):
        #For 1st variable (a), values repeat after one by one till its len() i.e. len(a).
        #For 2nd variable, values repeat after len() of its previous variable i.e. len(a)
        #For 3rd variable, values repeat after len(a)*len(b). 
    '''    for i in range(len(ds)):
            #if variable's index is 0, i.e 1st one is variable (a)
            if which_is_variable == 0:
                if i < pos_arr[0]:
                    mixer.append(ds[i])
            #if variable's index is 1, i.e 2nd one is variable (b)
            if which_is_variable == 1:
                if i < pos_arr[0]*pos_arr[1]:
                    mixer = ds[0:pos_arr[0]*pos_arr[1]-1:pos_arr[0]]
            #if variable's index is 2, i.e 3rd one is variable (c)
            if which_is_variable == 2:
                if i < pos_arr[0]*pos_arr[1]*pos_arr[2]:
                    mixer = ds[0:pos_arr[0]*pos_arr[1]*pos_arr[2]-1:pos_arr[0]*pos_arr[1]]
            #if variable's index is 3, i.e 4th one is variable (d)
            if which_is_variable == 3:
                if i < pos_arr[0]*pos_arr[1]*pos_arr[2]*pos_arr[3]:
                    mixer = ds[0:pos_arr[0]*pos_arr[1]*pos_arr[2]*pos_arr[3]-1:pos_arr[0]*pos_arr[1]*pos_arr[2]]
    '''
    #print(mixer)
    
    Press, Mix_molec = [], []
    gridlist = []
    griddict = {}
    if posq == 'fc':
        pos_seq = [t, p, co, mett]
    elif posq == 'poseidon':
        pos_seq = [mett, co, t, p]
    #Let's make the grid dictionary
    for i in pos_seq[3]:
        for j in pos_seq[2]:
            for k in pos_seq[1]:
                for l in pos_seq[0]:
                    gridlist.append((l,k,j,i))
    #print(gridlist)
    for i in range(len(gridlist)):
        griddict[gridlist[i]] = ds[i]
    # print(griddict)
    
    for i,j in griddict.items():
        if posq == 'fc':
            if i[0] == fixed_T and i[2] == fixed_co and i[3] == fixed_met:
                # print(i,' : ',j[spl.index(molecule)])
                Press.append(i[which_is_variable])
                Mix_molec.append(j)
        elif posq == 'poseidon':
            if i[2] == fixed_T and i[1] == fixed_co and i[0] == fixed_met:
                # print(i,' : ',j[spl.index(molecule)])
                Press.append(i[which_is_variable])
                Mix_molec.append(j)
            
    return Mix_molec, Press

Mix_molec, Press = h5read(file_fc,which_molc, fixed_T, fixed_co, fixed_met, pos_tuple, mix_name, 'fc')
plt.loglog(Mix_molec,Press,'--',label = 'FastChem grid',color='blue')

# Mixx, Press_d = h5read(file_pos,'H2O',fixed_T, fixed_co, fixed_met,['Met', 'C_O', 'T', 'P'],'log(X)','poseidon')
# print(Mixx,Press_d)
# Mixxx = [10**i for i in Mixx]
# plt.loglog(Mixxx, Press_d,'--',label='Poseidon grid',color='green')

#Poseidon_read
def read_fastchem_grid(chem_species, input_path):
    """
    Reads the FastChem grid from the specified HDF5 file and loads the data for the given chemical species.

    Parameters:
    - chem_species: list of chemical species to load from the grid.
    - input_path: path to the HDF5 file containing the FastChem grid data.

    Returns:
    - fastchem_grid: dictionary containing the loaded grid data.
    """
    # Open the HDF5 file
    data = h5py.File(input_path, 'r')
    
    # Load the grids from the HDF5 file
    T_grid = np.array(data['Info/T grid'])
    P_grid = np.array(data['Info/P grid'])
    Metal_grid = np.array(data['Info/M/H grid'])
    c_to_o_grid = np.array(data['Info/C/O grid'])
    
    # Initialize the log_X_grid array to store the chemical species data
    log_X_grid = np.zeros((len(chem_species), len(Metal_grid), len(c_to_o_grid),
                           len(T_grid), len(P_grid)))
    
    # Load the data for each chemical species
    for q, chem in enumerate(chem_species):
        raw_array = np.array(data[f'{chem}/log(X)'])
        reshaped_array = raw_array.reshape(len(Metal_grid), len(c_to_o_grid),
                                           len(T_grid), len(P_grid))
        log_X_grid[q, :, :, :, :] = reshaped_array
    
    # Close the HDF5 file
    data.close()
    
    # Create a dictionary to store the loaded grid data
    fastchem_grid = {
        'log_X_grid': log_X_grid,
        'T_grid': T_grid,
        'P_grid': P_grid,
        'Metal_grid': Metal_grid,
        'c_to_o_grid': c_to_o_grid
    }
    
    return fastchem_grid, Metal_grid, c_to_o_grid, T_grid, P_grid
pos_grid, Metal_grid, c_to_o_grid, T_grid, P_grid = read_fastchem_grid([pos_molc], input_path=file_pos)
# print(pos_grid)
mix_x = pos_grid['log_X_grid'][0,np.where(Metal_grid==fixed_met)[0],np.where(c_to_o_grid==fixed_co)[0],np.where(T_grid==fixed_T)[0],:]
mix_x_f = mix_x[0]
mix_x_f = [10**i for i in mix_x_f]
press_y = pos_grid['P_grid']
print(mix_x_f,press_y)
plt.loglog(mix_x_f, press_y, '--',label='Poseidon grid',color='green')
# plt.loglog(mixer,p,'--',label = 'FastChem grid')

#PyChem output
#Remove blank spaces in PyChem output # and Pressure.
want_pychem = 'y'
if want_pychem == 'y':
    pych = '/home/pc6/Downloads/pychem_1500K.dat'
    read_pyc = pd.read_csv(pych, sep= '\s+', skiprows=0)

    coloms = list(read_pyc.columns)
    Tval = fixed_T
    COval = fixed_co
    metval = fixed_met
    Pval = read_pyc.iloc[:,0]
    
    def formulae(x):
        if x == 'HCN':
            return 'C1H1N1'
        
        else:
            return x
        
    pyc_mix = which_molc
    mix_pyc = read_pyc.iloc[:,coloms.index(formulae(pyc_mix))]

    #print(Pval,mix_pyc)

    plt.loglog(mix_pyc,Pval,'--',label='PyChem grid',color='red')
else:
    pass



plt.gca().invert_yaxis()
plt.xlabel('VMR')
plt.ylabel('P (bar)')
# plt.xlim([1e-8,1e0])
plt.title(f'Species: {which_molc} at T = {fixed_T} K, C/O = {fixed_co} and [Fe/H] = {fixed_met}')
plt.legend()
plt.show() 
       