import pyfastchem
import numpy as np
import os
from save_output import saveChemistryOutputPandas
import matplotlib.pyplot as plt
from astropy import constants as const
import h5py
from molmass import Formula

#Modify the input file

spec_choose = ['C','H','He','O','N','P','S','Ti','V','Si'] #,'Al','Ar','Ca','Cl','Co','Cr','Cu','F','Fe','Ge','K','Mg','Mn','Na','Ne','Ni','Zn']

inp_path = '/home/pc6/FastChem/input/element_abundances/asplund_2020.dat'
out_path = '/home/pc6/FastChem/input/element_abundances/asplund_2020_to_nexotrans.dat'
logK_path = '/home/pc6/FastChem/input/logK/logK.dat'
value_abs = '-60' #Value of abundance of molecules that you don't want in FastChem. Avoid 0, use any negative integer of higher magnitude. 
want_electron = 'n' #Set as 'n' if you don't want electrons (and ions) in FastChem output, else set to 'y' or anything else. 
remove_others = 'y' #Set as 'y' if you want to remove all other elements except the chosen ones. Else set 'n' or anything else. 
 

# Open the input file for reading
with open(inp_path, 'r') as infile:
    lines = infile.readlines()
    

# Open the output file for writing
with open(out_path, 'w') as outfile:
    for line in lines:
        # Ignore comment lines
        if line.startswith('#'):
            outfile.write(line)
            continue
        
        # Split the line into element and value
        parts = line.split()
        if len(parts) == 2:
            element, value = parts[0], parts[1]
            # Check if the element is present in the list, if not replace it with -60
            if want_electron == 'n':
                if element == 'e-':
                    continue
            
            if element not in spec_choose:
                value = value_abs
            
            if remove_others == 'y':
                if element not in spec_choose:
                    continue
                
            outfile.write(f"{element}  {value}\n")
        else:
            outfile.write(line)  # Write the line as it is if it's not in the expected format


#After this, remove the electron row entirely from the file if not present, don't make e- abundance 0.

######################

#temperature_values = np.linspace(200,4000, 39)
#pressure_values = np.logspace(-7, 2, num=28)
#c_to_o_values = np.linspace(0.2, 2.0, 25)
#metallicity_values = np.logspace(-1, 3, 53)

temperature_values = np.linspace(300,4000, 38)
print('T grid: ',temperature_values)
pressure_values = np.logspace(-7, 2, 19)
print('P grid: ',pressure_values)
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
metallicity_values = np.logspace(-1, 4, 101)


# Define the directory for the output
output_dir = '/home/pc6/FastChem/output_pyfc/trials_nexotrans'

# Create a FastChem object
fastchem = pyfastchem.FastChem(
    out_path,
    logK_path,
    1)
 
fastchem.setParameter('nbIterationsChem', 200000)
fastchem.setParameter('accuracyChem', 1e-4)   
#fastchem.setParameter('nbSwitchToNewton', 400)
#fastchem.setParameter('nbIterationsNewton', 5000)

# Allocate the data for the output
# totalnumber of datapoints
nb_points = len(temperature_values) * len(pressure_values) * len(c_to_o_values) * len(metallicity_values)


number_densities = np.zeros((nb_points, fastchem.getGasSpeciesNumber()))
total_element_density = np.zeros(nb_points)
mean_molecular_weight = np.zeros(nb_points)
element_conserved = np.zeros((nb_points, fastchem.getElementNumber()), dtype=int)
fastchem_flags = np.zeros(nb_points, dtype=int)
nb_iterations = np.zeros(nb_points, dtype=int)
nb_chemistry_iterations = np.zeros(nb_points, dtype=int)
nb_cond_iterations = np.zeros(nb_points, dtype=int)

temperature = np.zeros(nb_points)
pressure = np.zeros(nb_points)
c_to_o = np.zeros(nb_points)
metallicity = np.zeros(nb_points)

# Make a copy of the solar abundances from FastChem
solar_abundances = np.array(fastchem.getElementAbundances())
#print(solar_abundances)
# We need to know the indices for O and C from FastChem
index_C = fastchem.getElementIndex('C')
index_O = fastchem.getElementIndex('O')

# Generate meshgrids for temperature, pressure, C/O ratio, and metallicity
# T_grid, P_grid, c_to_o_grid, Metal_grid = np.meshgrid(temperature_values, pressure_values,
                                                      # c_to_o_values, metallicity_values, indexing='ij')
# grid_shape = T_grid.shape
chem_species_len = fastchem.getGasSpeciesNumber()

# Allocate the grid for log mixing ratios
# mix_grid = np.zeros((chem_species_len,) + grid_shape)

#Calculate total metal abundance excluding H and He
# total_metal_abundance_solar = np.sum(solar_abundances[2:])

mixing_ratios_grid = {}
# Loop over the temperature, pressure, C/O ratios, and metallicity values
index = 0

for met in range(len(metallicity_values)):
    for ratio in range(len(c_to_o_values)):
        for press in range(len(pressure_values)):
            for temp in range(len(temperature_values)):
                #print(str(index+1)+'/'+str(nb_points))
                print(f"\rIteration: {index+1}/{nb_points}", end="")
                element_abundances = np.copy(solar_abundances)
                
                # Adjust the C and O abundances to maintain the C/O ratio
                # total_C_O = element_abundances[index_C] + element_abundances[index_O]
                # element_abundances[index_C] = total_C_O * (ratio / (1 + ratio))
                # element_abundances[index_O] = total_C_O * (1 / (1 + ratio))
                
                # Calculate remaining metals' abundance
                # total_remaining_metal_abundance = total_metal_abundance_solar * met
                
                # Scale the element abundances except those of H and He\
                
                for j in range(0, fastchem.getElementNumber()):
                    if fastchem.getElementSymbol(j) != 'H' and fastchem.getElementSymbol(j) != 'He':
                        element_abundances[j] *= metallicity_values[met]
                
                element_abundances[index_C] = element_abundances[index_O] * c_to_o_values[ratio]
                
                fastchem.setElementAbundances(element_abundances)
                
                # Create the input and output structures for FastChem
                input_data = pyfastchem.FastChemInput()
                output_data = pyfastchem.FastChemOutput()
                
                input_data.temperature = [temperature_values[temp]]
                input_data.pressure = [pressure_values[press]]
                
                fastchem_flag = fastchem.calcDensities(input_data, output_data)
                
                # Copy the FastChem input and output into the pre-allocated arrays
                temperature[index] = input_data.temperature[0]
                pressure[index] = input_data.pressure[0]
                # c_to_o[index] = ratio
                # metallicity[index] = met
                
                number_densities[index, :] = np.array(output_data.number_densities[0])
                total_element_density[index] = output_data.total_element_density[0]
                mean_molecular_weight[index] = output_data.mean_molecular_weight[0]
                element_conserved[index, :] = output_data.element_conserved[0]
                fastchem_flags[index] = output_data.fastchem_flag[0]
                nb_iterations[index] = output_data.nb_iterations[0]
                nb_chemistry_iterations[index] = output_data.nb_chemistry_iterations[0]
                nb_cond_iterations[index] = output_data.nb_cond_iterations[0]
                
                # Calculate the gas number density from the ideal gas law
                gas_number_density = pressure_values[press] * 1e6 / (const.k_B.cgs * temperature_values[temp])
                
                # Calculate mixing ratios and convert to log10
                mixing_ratios = np.array(output_data.number_densities[0]) / gas_number_density.value
                
                # if index == 2:
                #     print(mixing_ratios,len(mixing_ratios),chem_species_len)
                # if 0 in mixing_ratios:
                #     print('hello')
                
                # if ratio == 0 and met ==0 and press ==0 and temp ==0:
                #     if 0 in mixing_ratios:
                #         zp = np.where(mixing_ratios == 0)
                #         z = zp[0][0]
                #         print(' ',z, ' ')
                #         print(output_data.number_densities[0][z],gas_number_density.value)
                ##log_mixing_ratios = np.log10(mixing_ratios)
                # if np.nan in log_mixing_ratios or -np.inf in log_mixing_ratios:
                #     print('hello')
                
                # Store the mixing ratios in the dictionary with (T, P, C/O, metallicity) as the key
                grid_point = (temperature_values[temp], pressure_values[press], c_to_o_values[ratio], metallicity_values[met])
                mixing_ratios_grid[grid_point] = mixing_ratios
                
                index += 1
print(f"\rIteration: {index}/{nb_points}")
species_list = []
for ind in range(len(mixing_ratios)):
    species_list.append(fastchem.getGasSpeciesSymbol(ind))
# for k, v in mixing_ratios_grid.items():
#     print(k)
                
#element_abundances
#elem= element_conserved

# Convergence summary report
print("FastChem reports:")
print("  -", pyfastchem.FASTCHEM_MSG[np.max(fastchem_flags)])

if np.amin(output_data.element_conserved) == 1:
    print("  - element conservation: ok")
else:
    print("  - element conservation: fail")
    
# Total gas particle number density from the ideal gas law 
gas_number_density = pressure * 1e6 / (const.k_B.cgs * temperature)

# Check if output directory exists, create it if it doesn't
os.makedirs(output_dir, exist_ok=True)

# Prepare additional columns and descriptions
additional_columns = np.vstack([c_to_o, metallicity])
additional_columns_desc = ['C/O', 'M/H']

# Save the output as a pandas DataFrame inside a pickle file
saveChemistryOutputPandas(output_dir + '/chemistry.pkl', 
                          temperature, pressure,
                          total_element_density, 
                          mean_molecular_weight, 
                          number_densities,
                          fastchem, 
                          None, 
                          additional_columns, 
                          additional_columns_desc)

from save_output import  saveMonitorOutput

saveMonitorOutput(output_dir + '/monitor.dat', 
                  temperature, pressure, 
                  element_conserved,
                  fastchem_flags,
                  nb_iterations,
                  nb_chemistry_iterations,
                  nb_cond_iterations,
                  total_element_density,
                  mean_molecular_weight,
                  fastchem,
                  additional_columns, 
                  additional_columns_desc)

# import pickle
# with open(output_dir + '/chemistry.pkl', 'rb') as f:
#     data = pickle.load(f)
    
# Save the data to an HDF5 file
out_path = os.path.join(output_dir, 'full_grid_tiv_exact_poseidon_asp2020_fullgrid.h5')
# if 'C1H1N1_1' in species_list:
#     species_list[species_list.index('C1H1N1_1')] = 'HCN'
# if 'C1H1N1_2' in species_list:
#     species_list[species_list.index('C1H1N1_2')] = 'HNC'
def formulae(x):
    if x == 'C1H1N1_1':
        return 'HCN'
    elif x == 'C1H1N1_2':
        return 'HNC'
    elif x == 'C1N2_cnn':
        return 'C1N2cnn'
    elif x == 'C1N2_ncn':
        return 'C1N2ncn'
    elif x == 'H1N1O2cis':
        return 'H1N1O2c'
    elif x == 'H1N1O2trans':
        return 'H1N1O2t'
    
    else:
        return x
# print(species_list)
# for i in range(len(species_list)):
#     print(formulae(species_list[i]))
print('FastChem output saved. Creating .h5 file...')
specs_list = [formulae(species_list[i]) for i in range(len(species_list))]
with h5py.File(out_path, 'w') as hdf_file:
    hdf_file.create_dataset('Info/T grid', data=temperature_values)
    hdf_file.create_dataset('Info/P grid', data=pressure_values)
    hdf_file.create_dataset('Info/M/H grid', data=metallicity_values)
    hdf_file.create_dataset('Info/C/O grid', data=c_to_o_values)
    
    for i in range(len(species_list)):
        hdf_file.create_dataset(f'{specs_list[i]}/MR', data=[v[i] for v in mixing_ratios_grid.values()])

print('HDF5 file creation successful!')

param_tuple = ['T','P','C_O','Met']
fix_T = 300
# fix_P = 
fix_co = 0.2
fix_met = 1e-1
molecule = input(f'This is the list of species:\n{specs_list}\nWhich one you want to plot? Enter species name: ')
variable_par = 'P' #Should be one item from param_tuple. 
Press, Mix_molec = [], []
for i,j in mixing_ratios_grid.items():
    if i[0] == fix_T and i[2] == fix_co and i[3] == fix_met:
        # print(i,' : ',j[spl.index(molecule)])
        Press.append(i[param_tuple.index(variable_par)])
        Mix_molec.append(j[specs_list.index(molecule)])
plt.loglog(Mix_molec,Press,'--')
plt.gca().invert_yaxis()
plt.show()
