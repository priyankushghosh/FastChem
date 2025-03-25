# We aim to create NexoChem_beta grid for NexoTrans, but we will also create FastChem grids for benchmarking with exactly same parameters.
# import pyfastchem
import numpy as np
import os
# from save_output import saveChemistryOutputPandas
import matplotlib.pyplot as plt
from astropy import constants as const
import h5py
from molmass import Formula
from numba import njit


#Modify the input file for Fastchem

spec_choose = ['C','H','He','O','N','P','S','Ti','V','Si'] #,'Al','Ar','Ca','Cl','Co','Cr','Cu','F','Fe','Ge','K','Mg','Mn','Na','Ne','Ni','Zn']

inp_path = '/home/pc6/FastChem/input/element_abundances/asplund_2009.dat'
out_path = '/home/pc6/FastChem/input/element_abundances/asplund_2009_to_nexotrans.dat'
logK_path = '/home/pc6/FastChem/input/logK/logK.dat'
#value_abs = '-60' #Value of abundance of molecules that you don't want in FastChem. Avoid 0, use any negative integer of higher magnitude. 
#want_electron = 'n' #Set as 'n' if you don't want electrons (and ions) in FastChem output, else set to 'y' or anything else. 
#remove_others = 'y' #Set as 'y' if you want to remove all other elements except the chosen ones. Else set 'n' or anything else. 
 

temperature_values = np.linspace(300,4000, 38) #20)
# print('T grid: ',temperature_values)
pressure_values = np.logspace(-7, 2, 19) #20)
# print('P grid: ',pressure_values)
# c_to_o_values = np.linspace(0.2, 2.0, 4) #10)
c_to_o_values = np.array([2.000e-1,
2.500e-1,
3.000e-1,
3.500e-1,
4.000e-1,
4.500e-1,
5.000e-1,
5.500e-1,
6.000e-1,
6.500e-1,
7.000e-1,
7.500e-1,
8.000e-1,
8.500e-1,
9.000e-1,
9.095e-1,
9.190e-1,
9.286e-1,
9.381e-1,
9.476e-1,
9.571e-1,
9.667e-1,
9.762e-1,
9.857e-1,
9.952e-1,
1.005e+0,
1.014e+0,
1.024e+0,
1.033e+0,
1.043e+0,
1.052e+0,
1.062e+0,
1.071e+0,
1.081e+0,
1.090e+0,
1.100e+0,
1.169e+0,
1.238e+0,
1.308e+0,
1.377e+0,
1.446e+0,
1.515e+0,
1.585e+0,
1.654e+0,
1.723e+0,
1.792e+0,
1.862e+0,
1.931e+0,
2.000e+0])
#np.linspace(0.2, 2.0, 15)#49)
metallicity_values = np.logspace(-1, 3, 101) #10)

##################### NexoChem Inputs ##########################
#Append NexoChem path
import sys
path_nexochem = '/home/pc6/Downloads/NexoChem_beta'
# sys.path.append(path_nexochem)

# import nexochem as nxc
# #check if nxochem.py exists
# if not os.path.exists(path_nexochem+'/nexochem.py'):
#     raise Exception('nexochem.py not found in the path: ',path_nexochem)

#### Full NexoChem code ############
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import constants as const
import sys

# Get the absolute path to the 'input' directory
runn_dir = path_nexochem+'/src' #os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input'))
input_dir = path_nexochem+'/input'
# Add the 'input' directory to the system path
sys.path.append(runn_dir)
sys.path.append(input_dir)
# os.system(f'python3 {input_dir}/nexotrans_janaf_data_updated.py')
from nexotrans_janaf_data_updated import *
from scipy.optimize import minimize
# import cvxpy as cp
import time
from scipy.linalg import solve
import numba 
import multiprocessing as mp
from joblib import Parallel, delayed
# %matplotlib qt
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

star_time = time.time()

############################## INPUTS ###############################
#primary inputs

# temperature[:] = 500


#secondary input
nsteps = 200000 #No. of iteration for convergence of Gibbs minimizer
ncore = -2 #The number of cores you want to use, -1 means all cores, -2 means all cores except 1 available for other tasks. Recommended: -2 or -3.
# meta = 10 
# c_o = '10' #'d' for default case

#the chemical species we want to plot later
#note that the standard FastChem input files use the Hill notation
# plot_species =['H2O1', 'C1O2', 'C1O1','O2', 'H2C2','C1H4','H3N1','C2F2']
plot = 'no'

#####################################################################

#=============================================================================================================


# p_t_data = np.loadtxt('../input/example_p_t_structures/jupiter.dat', skiprows=1)

# pressure = p_t_data[:,0]
# temperature = p_t_data[:,1] 
# temperature[:] = 500

#some input values for temperature (in K) and pressure (in bar)

                                                                                                        

# pressure = np.array([  3.16227770e+01,   2.63026800e+01,   2.18776160e+01,
#          1.81970090e+01,   1.51356120e+01,   1.25892540e+01,
#          1.04712850e+01,   8.70963590e+00,   7.24435960e+00,
#          6.02559590e+00,   5.01187230e+00,   4.16869380e+00,
#          3.46736850e+00,   2.88403150e+00,   2.39883290e+00,
#          1.99526230e+00,   1.65958690e+00,   1.38038430e+00,
#          1.14815360e+00,   9.54992590e-01,   7.94328230e-01,
#          6.60693450e-01,   5.49540870e-01,   4.57088190e-01,
#          3.80189400e-01,   3.16227770e-01,   2.63026800e-01,
#          2.18776160e-01,   1.81970090e-01,   1.51356120e-01,
#          1.25892540e-01,   1.04712850e-01,   8.70963590e-02,
#          7.24435960e-02,   6.02559590e-02,   5.01187230e-02,
#          4.16869380e-02,   3.46736850e-02,   2.88403150e-02,
#          2.39883290e-02,   1.99526230e-02,   1.65958690e-02,
#          1.38038430e-02,   1.14815360e-02,   9.54992590e-03,
#          7.94328230e-03,   6.60693450e-03,   5.49540870e-03,
#          4.57088190e-03,   3.80189400e-03,   3.16227770e-03,
#          2.63026800e-03,   2.18776160e-03,   1.81970090e-03,
#          1.51356120e-03,   1.25892540e-03,   1.04712850e-03,
#          8.70963590e-04,   7.24435960e-04,   6.02559590e-04,
#          5.01187230e-04,   4.16869380e-04,   3.46736850e-04,
#          2.88403150e-04,   2.39883290e-04,   1.99526230e-04,
#          1.65958690e-04,   1.38038430e-04,   1.14815360e-04,
#          9.54992590e-05,   7.94328230e-05,   6.60693450e-05,
#          5.49540870e-05,   4.57088190e-05,   3.80189400e-05,
#          3.16227770e-05,   2.63026800e-05,   2.18776160e-05,
#          1.81970090e-05,   1.51356120e-05,   1.25892540e-05,
#          1.04712850e-05,   8.70963590e-06,   7.24435960e-06,
#          6.02559590e-06,   5.01187230e-06,   4.16869380e-06,
#          3.46736850e-06,   2.88403150e-06,   2.39883290e-06])

# temperature = np.array([ 1811.8938 ,  1810.9444 ,  1810.1535 ,  1809.4948 ,  1808.9463 ,
#         1808.4898 ,  1808.1098 ,  1807.7936 ,  1807.5304 ,  1807.3114 ,
#         1807.1291 ,  1806.9766 ,  1806.8464 ,  1806.7212 ,  1806.53   ,
#         1806.1269 ,  1805.2849 ,  1803.7403 ,  1800.5841 ,  1794.9518 ,
#         1786.6255 ,  1775.3705 ,  1761.0973 ,  1742.3631 ,  1719.6396 ,
#         1694.0976 ,  1666.1517 ,  1636.2055 ,  1603.0265 ,  1567.6227 ,
#         1531.3326 ,  1494.5529 ,  1457.6432 ,  1419.5923 ,  1381.4921 ,
#         1344.4864 ,  1308.8483 ,  1274.7949 ,  1241.6997 ,  1210.3302 ,
#         1181.2851 ,  1154.5844 ,  1130.2066 ,  1107.7735 ,  1087.5396 ,
#         1069.5885 ,  1053.7529 ,  1039.8606 ,  1027.6493 ,  1017.057  ,
#         1007.9573 ,  1000.1681 ,   993.52384,   987.85759,   983.05871,
#          979.01311,   975.60886,   972.74937,   970.3489 ,   968.33983,
#          966.66146,   965.26054,   964.09216,   963.11826,   962.30738,
#          961.63264,   961.0714 ,   960.60473,   960.2169 ,   959.89468,
#          959.627  ,   959.40467,   959.22003,   959.06677,   958.93955,
#          958.83394,   958.74627,   958.6735 ,   958.61313,   958.56303,
#          958.52146,   958.48696,   958.45833,   958.43458,   958.41487,
#          958.39852,   958.38496,   958.3737 ,   958.36437,   958.35662])
# # temperature[:]=300

# plot_species_symbols = plot_species


########################### Define NexoChem functions ######################

def elem_md(elem_abun):
   '''
   OUTPUT:
   md = fraction of elemental abundance with respect to hydrogen
   '''
   #elem abun  = ln(fraction) +12
   md = 10**(elem_abun.astype(np.float64) - 12)
   return md


# g0/RT (standard chemical potential calculator) using fitting coefficients
# go/RT = a1/T + a2*ln(T) + a3 + a4*T + a5*(T**2)

def delta_g_calc(molno,T):
   '''
   INPUTS:
   molno = index of molecule in molecules array
   T = Temperature
   OUTPUT:
   lnk_arr = array of go/RT for all moecules at temperature T
   '''
   lnk_arr = [fit[i,0]/T + fit[i,1]*np.log(T) + fit[i,2] + fit[i,3]*T + fit[i,4]*(T**2) for i in molno]
   return lnk_arr 

# Function to calculate inital molar density
# The aim is to calulate a non zero initial moelcuar abundace which satifies mass balance equation Ni*AIJ = Bj
# This is done by having as many free variables as the number of elements and other fixed variables

def ini_balance(b):
   '''
   INPUT:
   b = elemental ratio with respect to H # this has been taking from the aspund_2009 solar abundance data 
   OUTPUT:
   x_ini = intial molecular molar density 
   '''
   # Start with initial arrays of ones
   x_ini = np.ones(len(Aij) , np.double)
   
   # Number of elements
   ne = len(elements)
   # free variable array filled with negative ones
   xnew = -np.ones(ne)
   # Keep updating values till all vlaues are positive

   while (np.any(xnew[xnew<=0])):
       # If negative values are still present the fixed varaibles are scaled by a factor of 0.1
       x_ini*=0.1
       # Using martix algebra to solve for mass balance equation
       A1 = np.dot(Aij.T[:,ne+1:],x_ini[ne+1:])
       b_new = b - A1
       xnew = np.linalg.solve(Aij.T[:,:ne], b_new)

   # Update the whole intial abundances
   x_ini[:ne] = xnew
   # print(x_ini)
   return(x_ini)   

# def dfdl(lam,delta_i , yi,ci):
#    df_dl = delta_i*(ci+ np.log((yi + lam*delta_i))-np.log((np.sum(yi)-lam*(np.sum(delta_i)))))
#    return np.sum(df_dl)

def fyi(yi,ci): 
   fiy = yi*(ci+np.log(yi/(np.sum(yi))))
   return fiy

@njit
def afyi(fiy):
   afiy = np.dot(Aij.T,fiy)
   afiy  =np.concatenate((afiy,np.array([np.sum(fiy)])))
   return afiy

def Lambda_correction1(yi, ci, delta_i, lamda):
    found = True
    
    # Compute valid terms for logarithms
    log_term1_valid = yi[:, np.newaxis] + lamda * delta_i[:, np.newaxis]
    log_term2_valid = np.sum(yi) - lamda * np.sum(delta_i)
    
    # Mask to filter out invalid values
    valid_mask1 = log_term1_valid > 0
    valid_mask2 = log_term2_valid > 0
    
    # Initialize arrays to avoid issues with invalid values
    log_term1 = np.full_like(log_term1_valid, np.nan, dtype=np.float64)
    log_term2 = np.nan
    
    # Compute logarithms with masking
    log_term1[valid_mask1] = np.log(log_term1_valid[valid_mask1])
    if valid_mask2.any():  # Check if there are any valid values
        log_term2 = np.log(log_term2_valid)
    
    # Compute F with masked values
    F = delta_i[:, np.newaxis] * (ci[:, np.newaxis] + log_term1 - log_term2)
    F = np.sum(F, axis=0)
    
    # Find negative values in F
    if np.any(F < 0):   
        lam_val_m = lamda[F < 0][-1]
        found = False
    
    xi_final = yi if found else yi + lam_val_m * delta_i
    
    return xi_final

def main(TP, yi, md, index):
    lamda = np.concatenate([np.exp(np.linspace(-50, np.log(1), 100))]) #Change at your convenience
    # print("pressure and Temperature: ", TP[0], TP[1])
    # print("\n")
    ci = np.array(delta_g_calc(index, TP[0])) + np.log(TP[1])
   #  lamda = np.concatenate([np.exp(np.linspace(-100, np.log(0.5), 100)), np.arange(0.5, 1, 0.1)])
    
    prev_yi = yi.copy()  # Store the initial yi
    
    for j in range(5000):
        fiy = fyi(yi, ci)
        A = afyi(fiy)
        Ay = np.array([Aij[i] * yi[i] for i in range(len(Aij))], dtype=np.float64)
        R = np.dot(Aij.T, Ay)
        Rij = np.zeros((len(R) + 1, len(R) + 1))
        for i in range(len(md)):
            Rij[i, -1] = md[i]
            Rij[-1, i] = md[i]
        Rij[:-1, :-1] = R
        Sol = np.dot(np.linalg.inv(Rij), A)
        xi = -fiy + yi * (Sol[-1] + 1) + np.dot(Ay, Sol[:-1])
        if np.any(xi <= 0):
            xi = Lambda_correction1(yi, ci, xi - yi, lamda)
        diff = xi * np.sum(yi) / (np.sum(xi) * yi) - 1
        
        if np.sum(np.abs(diff)) / len(diff) < 1e-4:
            break
        
        # Check for NaN values in xi
        if np.any(np.isnan(xi)):
            print(f"NaN encountered at iteration {j} of pressure{TP[1]} and temperature{TP[0]}, breaking the loop")
            yi = prev_yi  # Revert to the previous yi
            break
        
        prev_yi = yi.copy()  # Update prev_yi before updating yi
        
        yi = xi

    return yi
   # Number of elements

def parallel_main(TP, yi, md, index):
    return main(TP, yi, md, index)

def formulae_nx(x):
    if x == 'C1H1N1':
        return 'HCN'
    # elif x == 'C1H1N1_2':
    #     return 'HNC'
    # elif x == 'C1N2_cnn':
    #     return 'C1N2cnn'
    # elif x == 'C1N2_ncn':
    #     return 'C1N2ncn'
    # elif x == 'H1N1O2cis':
    #     return 'H1N1O2c'
    # elif x == 'H1N1O2trans':
    #     return 'H1N1O2t'
    
    else:
        return x

############################################################################

k = 1.380649e-16 #cgs

pressure = pressure_values #np.logspace(-8,-2,num =100)
temperature = np.zeros(len(pressure))
out_pathe = '/home/pc6/FastChem/output_pyfc/NexoChem_beta_nexotrans.h5'
spex_list = [formulae_nx(molecules[i]) for i in range(len(molecules))]
# for i in spex_list: #Removing duplicates from the list
#     if i in list(set([item for item in spex_list if spex_list.count(item) > 1])):
#         spex_list.remove(i)
# print(molecules)

indexx = 0
totg_points = len(metallicity_values)*len(c_to_o_values)*len(pressure)*len(temperature_values)

#The big loop for NexoChem
    
net_mix_grid = {}

for m in range(len(metallicity_values)):
    for c in range(len(c_to_o_values)):
        for k in range(len(pressure)):
            for l in range(len(temperature_values)):
                temperature[:] = temperature_values[l]
            
                gas_number_density = pressure[k]*1e6 / (k * temperature)
                #==========================================================================================================
                mols = molecules

                # Calculates the elemental abundance with respect to hydrogen from dex(decimal exponent) form
                #check the google doc- code log for more explination 
                
                index = []

                for i in mols:
                    ind = np.where(molecules == i)[0][0]
                    index.append(ind) 


                # lamda = np.concatenate([np.exp(np.linspace(-50, np.log(1), 100))])
                #lamda = np.concatenate([np.exp(np.linspace(-50, np.log(0.5), 100)), np.arange(0.5, 1, 0.1)])


                h_inx = list(elements).index('H')

                he_inx = list(elements).index('He')
                md = elem_md(elem_abun)



                for j,i in enumerate(md):
                    if i == md[h_inx]:
                        pass
                        
                    elif i == md[he_inx]:
                        pass
                    else:
                        md[j] *= metallicity_values[m]

                c_inx = list(elements).index('C')
                o_inx = list(elements).index('O')

                
                md[c_inx] = float(c_to_o_values[c]) * md[o_inx]
                    
                yi = ini_balance(md)
                #print(yi)
                Ay = np.zeros_like(Aij,dtype=np.float64) 
                Aij_length = len(Aij)


                if __name__ == '__main__':
                    temperature = temperature  # your temperature array
                    pressure = pressure  # your pressure array
                    md = md  # your md array
                    yi = yi  # your yi array
                    index = index  # your index value

                    grid_val = (temperature_values[l], pressure_values[k], c_to_o_values[c], metallicity_values[m])
                    #print(grid_val)

                    # Using joblib to parallelize
                    sol = Parallel(n_jobs=ncore)(delayed(parallel_main)([temperature[i], pressure[i]], yi, md, index) for i in range(len(temperature)))

                    sol = np.array(sol) 

                # sol = np.column_stack((pressure, temperature, sol))

                #pychem_time = time.time()-star_time
                #------------------------------------------------------------------------------------------------------------------------

                header = ['Pressure', 'Temperature'] + molecules.tolist()

                # Create a new array with 12 columns
                output= np.zeros((len(pressure), len(molecules)+2))

                # Fill the first two columns with pressure and temperature
                output[:, 0] = pressure
                output[:, 1] = temperature

                # Fill the remaining columns 
                output[:, 2:] = sol

                mixingratios = sol[k, :] #1st index for pressure variable (T = constant), 2nd index for molecules with same order in molecules array.  

                #Store the mixing ratios in the dictionary with (T, P, C/O, metallicity) as the key
                grid_pt = (temperature_values[l], pressure_values[k], c_to_o_values[c], metallicity_values[m])
                net_mix_grid[grid_pt] = mixingratios
                
                
                print(f"\rIteration: {indexx+1}/{totg_points}", end="")
                #print(grid_pt)
                indexx += 1

# print(list(set([item for item in spex_list if spex_list.count(item) > 1])))
# print(spex_list,'\n')
# print(net_mix_grid)   
with h5py.File(out_pathe, 'w') as hdf_file:
    hdf_file.create_dataset('Info/T grid', data=temperature_values)
    hdf_file.create_dataset('Info/P grid', data=pressure_values)
    hdf_file.create_dataset('Info/M/H grid', data=metallicity_values)
    hdf_file.create_dataset('Info/C/O grid', data=c_to_o_values)
    for i in range(len(spex_list)):
        hdf_file.create_dataset(f'{spex_list[i]}/MR', data=[v[i] for v in net_mix_grid.values()])
    # plt.plot(sol[:,i],pressure,linestyle ='-',label =f"Pychem Species {i}")

# Create the formatted output
# with open('../output/output.dat', 'w') as f:
#     # Write the header
#     header_format = '{:<28}' * len(header)
#     f.write(header_format.format(*header) + '\n')

#     # Write the data
#     for row in output:
#         row_format = '{:<28.18e}' * len(row)
#         f.write(row_format.format(*row) + '\n')


# temperature_columns = output[:, 1] 

#=======================================================================================================================

# elements_to_find = plot_species  #['C1O1','C1O2', 'C1H4', 'H2O1', 'H3N1']
# colors = ['orange','blue','yellow','red','brown','green','purple','cyan','pink','black', 'gray', 'violet' ]


# Find the indices of the elements
# indices = np.where(np.isin(molecules, elements_to_find))
# indices = indices[0]

# ==========================================================================================================

# pychem_indices =indices

# if plot == 'yes':
#     for i in pychem_indices:
#         # plt.plot(sol[:,i],pressure,linestyle ='-',label =f"Pychem Species {i}")
#         #sol array has n columns which is total number of molecules present and 
#         #output array has 2 extra columns then sol array they are basically pressure and temperature 
#         #hence output array has (n+2) columns 
#         plt.plot(output[:,i+2],pressure,linestyle ='-',label =f"Pychem Species {i}")

#     plt.xscale('log')
#     plt.yscale('log')
#     plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

#     plt.xlabel("Mixing ratios")
#     plt.ylabel("Pressure (bar)")
#     plt.legend(plot_species_symbols)

#     plt.show()

# print("The shape of P-T profile is :", temperature.shape,"X", pressure.shape)
# print("Total time taken to run", pychem_time)

print("Total time taken by NexoChem: ", time.time()-star_time, " seconds")

