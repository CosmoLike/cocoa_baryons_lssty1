import os
import glob




# Config
sims = [f'Magneticum_C{i}' for i in range(1, 16) if i not in [8]]

sims += ['Magneticum_WMAP7']

# Settings
mask = 'M2' # M1 fid scale cut; M2 2.5' scale cut
probe = 'cosmic_shear'

# Define the directory and the original file
data_dir = "./"
orig_file = f"LSST_Y1_{probe}.dataset"

# Loop over the range 2 to 15
for sim in sims:
    # Read the original file
    with open(os.path.join(data_dir, orig_file), 'r') as file:
        orig_content = file.readlines()
    
    # Specify datavectorfile
    orig_content[0] = f'data_file = data_vector/lsst_y1_baryons_{sim}.modelvector\n'
    
    orig_content[2] = f'mask_file = masks/LSST_Y1_{probe}_{mask}.mask\n'
    
    orig_content = ''.join(orig_content)

    # Define the new file name
    new_file_name = f'LSST_Y1_emu_{probe}_{sim}_{mask}.dataset'
    print(new_file_name)
    # Write the new content to the new file
    with open(new_file_name, 'w') as file:
        file.write(orig_content)
