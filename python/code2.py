import pyfastchem
import numpy as np
import os
import h5py
from astropy import constants as const



# Mapping dictionary
species_mapping = {
    'H2O': 'H2O1',
    'CO2': 'C1O2',
    'CO': 'C1O1',
    'SO2': 'O2S1',
    #H2S
    #CH4
    #C2H2
    #C2H4
    
    # Add more mappings as needed
}

# Reverse mapping dictionary
reverse_species_mapping = {v: k for k, v in species_mapping.items()}

def convert_species_names_to_fastchem(species_list):
    return [species_mapping.get(species, species) for species in species_list]

def convert_species_names_to_common(species_list):
    return [reverse_species_mapping.get(species, species) for species in species_list]


#-----------------------------------------------------------------------------------------------------


# Input values for temperature (in K) and pressure (in bar)

temperature_values = np.linspace(200,4000, 38)
pressure_values = np.logspace(-7, 2, 28)
c_to_o_values = np.linspace(0.2, 2.0, 50)
metallicity_values = np.logspace(-1, 3, 50)

# Define the directory for the output
output_dir = '/home/seps05/fastchem/output_deka/trials'

# Create a FastChem object
fastchem = pyfastchem.FastChem(
    '/home/seps05/fastchem/input/element_abundances/asplund_2020 (copy).dat',
    '/home/seps05/fastchem/input/logK/logK.dat',
    1)

# Make a copy of the solar abundances from FastChem
solar_abundances = np.array(fastchem.getElementAbundances())

# We need to know the indices for O and C from FastChem
index_C = fastchem.getElementIndex('C')
index_O = fastchem.getElementIndex('O')

# Generate meshgrids for temperature, pressure, C/O ratio, and metallicity
T_grid, P_grid, c_to_o_grid, Metal_grid = np.meshgrid(temperature_values, pressure_values,
                                                      c_to_o_values, metallicity_values, indexing='ij')

grid_shape = T_grid.shape
chem_species_len = fastchem.getGasSpeciesNumber()

# Allocate the grid for log mixing ratios
log_X_grid = np.zeros((chem_species_len,) + grid_shape)

total_metal_abundance_solar = np.sum(solar_abundances[2:])

# Loop over each combination in the meshgrid and calculate mixing ratios
for idx, (T, P, c_to_o, Metal) in enumerate(zip(T_grid.ravel(), P_grid.ravel(), c_to_o_grid.ravel(), Metal_grid.ravel())):
    
    element_abundances = np.copy(solar_abundances)
        
    # Adjust the C and O abundances to maintain the C/O ratio
    total_C_O = element_abundances[index_C] + element_abundances[index_O]
    element_abundances[index_C] = total_C_O * (c_to_o / (1 + c_to_o))
    element_abundances[index_O] = total_C_O * (1 / (1 +c_to_o))
    
    total_remaining_metal_abundance = total_metal_abundance_solar * Metal
    
    # Scale the element abundances except those of H, He, C, and O
    for j in range(2, fastchem.getElementNumber()):
        #if j != index_C and j != index_O:
        element_abundances[j] *= total_remaining_metal_abundance / total_metal_abundance_solar
            
    fastchem.setElementAbundances(element_abundances)
    
    # Create the input and output structures for FastChem
    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()
    
    input_data.temperature = [T]
    input_data.pressure = [P]
    
    fastchem.calcDensities(input_data, output_data)
    
    # Calculate the gas number density from the ideal gas law
    gas_number_density = P * 1e6 / (const.k_B.cgs * T)
    
    # Calculate mixing ratios and convert to log10
    mixing_ratios = np.array(output_data.number_densities[0]) / gas_number_density.value
    log_mixing_ratios = np.log(mixing_ratios)
    
    # Store the log mixing ratios in the log_X_grid
    idx_tuple = np.unravel_index(idx, grid_shape)
    log_X_grid[:, idx_tuple[0], idx_tuple[1], idx_tuple[2], idx_tuple[3]] = log_mixing_ratios
    
   

# Convert FastChem species to common notation for saving
fastchem_species = [fastchem.getGasSpeciesSymbol(j) for j in range(chem_species_len)]
common_species_names = convert_species_names_to_common(fastchem_species)

# Save the data to an HDF5 file
output_path = os.path.join(output_dir, 'chemistry_grid_changed.h5')
with h5py.File(output_path, 'w') as hdf_file:
    hdf_file.create_dataset('Info/T grid', data=temperature_values)
    hdf_file.create_dataset('Info/P grid', data=pressure_values)
    hdf_file.create_dataset('Info/M/H grid', data=metallicity_values)
    hdf_file.create_dataset('Info/C/O grid', data=c_to_o_values)
    
    for i, species in enumerate(common_species_names):
        hdf_file.create_dataset(f'{species}/log(X)', data=log_X_grid[i])


#set the C abundance as a function of the C/O ratio
#element_abundances[index_C] = element_abundances[index_O] * ratio
