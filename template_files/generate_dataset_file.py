import os
import glob

# Settings
mask = 'M1' # M1 1 arcmin scale cut; M2 LSST Y1 default scale cut
probe = 'cosmic_shear'
arg1 = ''  # '', mead2015, mead2020; begin with `_` if not empty

# If True, set the datafile to use the EuclidEmulator2 datavector
# In this case DV is same for pca/hmcode
use_EE_datavector = False

# Define the directory and the original file
data_dir = "../data"



#----------------- Don't change anything below this line -----------------#
orig_file = f"LSST_Y1_{probe}.dataset"

if use_EE_datavector:
    assert len(arg1) == 0, "Can't use arg1 with EuclidEmulator2 datavector"

    # Only these cosmologies supported by EuclidEmulator2
    sims = [f'Magneticum_C{i}' for i in [7, 9, 11, 12]]
    sims += ['Magneticum_WMAP7']

else:
    sims = [f'Magneticum_C{i}' for i in range(1, 16) if i not in [8]]
    sims += ['Magneticum_WMAP7']


# Loop over the range 2 to 15
for sim in sims:
    # Read the original file
    with open(os.path.join(data_dir, orig_file), 'r') as file:
        orig_content = file.readlines()

    # Specify datavectorfile
    if use_EE_datavector:
        orig_content[0] = f'data_file = data_vector/lsst_y1_baryons_{sim}_EuclidEmu.modelvector\n'
        new_file_name = f'../data/LSST_Y1_emu_{probe}_{sim}_{mask}_EuclidEmu.dataset'

    else:
        orig_content[0] = f'data_file = data_vector/lsst_y1_baryons_{sim}{arg1}.modelvector\n'
        new_file_name = f'{data_dir}/LSST_Y1_emu_{probe}_{sim}_{mask}{arg1}.dataset'

    orig_content[2] = f'mask_file = masks/LSST_Y1_{probe}_{mask}.mask\n'

    orig_content = ''.join(orig_content)

    # Define the new file name
    print(new_file_name)
    # Write the new content to the new file
    with open(new_file_name, 'w') as file:
        file.write(orig_content)