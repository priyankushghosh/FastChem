import pyfastchem
import numpy as np
import os
from save_output import saveChemistryOutputPandas
import matplotlib.pyplot as plt
from astropy import constants as const

#temperature_values = np.linspace(200,4000, 39)
#pressure_values = np.logspace(-7, 2, num=28)
#c_to_o_values = np.linspace(0.2, 2.0, 25)
#metallicity_values = np.logspace(-1, 3, 53)

temperature_values = np.linspace(200,4000, 11)
pressure_values = np.logspace(-7, 2, 11)
c_to_o_values = np.linspace(0.2, 2.0, 5)
metallicity_values = np.logspace(-1, 3, 5)


# Define the directory for the output
output_dir = '/home/seps05/fastchem/output_deka/trials'


# Create a FastChem object
fastchem = pyfastchem.FastChem(
    '/home/seps05/fastchem/input/element_abundances/asplund_2020 (copy).dat',
    '/home/seps05/fastchem/input/logK/logK.dat',
    1)
 
fastchem.setParameter('nbIterationsChem', 100000)
fastchem.setParameter('accuracyChem', 1e-10)   
fastchem.setParameter('nbSwitchToNewton', 400)
fastchem.setParameter('nbIterationsNewton', 50000)

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

# We need to know the indices for O and C from FastChem
index_C = fastchem.getElementIndex('C')
index_O = fastchem.getElementIndex('O')

# Loop over the temperature, pressure, C/O ratios, and metallicity values
index = 0
for temp in temperature_values:
    for press in pressure_values:
        for ratio in c_to_o_values:
            for met in metallicity_values:
                
                element_abundances = np.copy(solar_abundances)
                
            
                # Adjust the C and O abundances to maintain the C/O ratio
                
                total_C_O = element_abundances[index_C] + element_abundances[index_O]
                element_abundances[index_C] = total_C_O * (ratio / (1 + ratio))
                element_abundances[index_O] = total_C_O * (1 / (1 + ratio))
                
            
                
                # Scale the element abundances except those of H and He
                
                for j in range(0, fastchem.getElementNumber()):
                    if fastchem.getElementSymbol(j) != 'H' and fastchem.getElementSymbol(j) != 'He':
                        element_abundances[j] *= met
                
                        
                fastchem.setElementAbundances(element_abundances)
                
                # Create the input and output structures for FastChem
                input_data = pyfastchem.FastChemInput()
                output_data = pyfastchem.FastChemOutput()
                
                input_data.temperature = [temp]
                input_data.pressure = [press]
                
                fastchem_flag = fastchem.calcDensities(input_data, output_data)
                
                
                # Copy the FastChem input and output into the pre-allocated arrays
                temperature[index] = temp
                pressure[index] = press
                c_to_o[index] = ratio
                metallicity[index] = met
                
                number_densities[index, :] = np.array(output_data.number_densities[0])
                total_element_density[index] = output_data.total_element_density[0]
                mean_molecular_weight[index] = output_data.mean_molecular_weight[0]
                element_conserved[index, :] = output_data.element_conserved[0]
                fastchem_flags[index] = output_data.fastchem_flag[0]
                nb_iterations[index] = output_data.nb_iterations[0]
                nb_chemistry_iterations[index] = output_data.nb_chemistry_iterations[0]
                nb_cond_iterations[index] = output_data.nb_cond_iterations[0]
                
                index += 1

element_abundances
elem= element_conserved

# Convergence summary report
print("FastChem reports:")
print("  -", pyfastchem.FASTCHEM_MSG[np.max(fastchem_flags)])

if np.amin(element_conserved) == 1:
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

#check the species we want to plot and get their indices from FastChem


#we could also save the figure as a pdf
#plt.savefig(output_dir + '/fastchem_c_to_o_fig.pdf')

import pickle
with open(output_dir + '/chemistry.pkl', 'rb') as f:
    data = pickle.load(f)
    

    
    
    