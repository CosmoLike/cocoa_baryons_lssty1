import os
from pathlib import Path
import yaml


def get_example_evaluate(flag_theory, probe, sim, mask, ARGS):
    if flag_theory:
        prefix = '_theory'
    else:
        prefix = ''
    
    template_file = yaml.safe_load(Path(f'{probe}_template.yaml').read_text())

    # Likelihood
    template_file['likelihood'][f'baryons_lsst_y1.lsst_{probe}']['data_file'] = f'LSST_Y1_emu_{probe}_{sim}_{mask}.dataset'


    # Emulator
    base_path = './projects/baryons_lsst_y1/'
    savedir =  f'y1_{probe}/{sim}/emu_output' # Used for saving chains
    emudir = f'/xdisk/timeifler/pranjalrs/baryons_lsst_y1_emu/y1_{probe}/{sim}/' # Used for saving training samples and emu

    if not os.path.isdir(savedir):
        print(f'Creating {savedir}...')
        os.makedirs(savedir)

    if not os.path.isdir(emudir):
        print(f'Creating {emudir}...')
        os.makedirs(emudir)

    emulator = template_file['emulator']

    emulator['io']['savedir'] = base_path + savedir
    emulator['io']['emudir'] = emudir
    emulator['io']['chainname'] = f'chain{prefix}{ARGS}_{mask}'

    emulator['training']['dv_fid'] = base_path + f'data/data_vector/lsst_y1_theory_{sim}.modelvector'
    emulator['sampling']['scalecut_mask'] = base_path + f'data/masks/LSST_Y1_{mask}_{probe}.mask'

    if not os.path.isdir(f'y1_{probe}/{sim}/'):
        print(f'Creating y1_{probe}/{sim}...')
        os.makedirs(f'y1_{probe}/{sim}/')
    
    print(f'y1_{probe}/{sim}/{sim}{prefix}{ARGS}_{mask}.yaml')
    with open(f'y1_{probe}/{sim}/{sim}{prefix}{ARGS}_{mask}.yaml', 'w') as yaml_file:
        yaml.dump(template_file, yaml_file, default_flow_style=False, sort_keys=False, line_break='\t')


## Specify yaml file settings
sims = [f'Magneticum_C{i}' for i in range(1,16) if i!=8]
sims += ['Magneticum_WMAP7']

for sim in sims:
    flag_theory = False # False for theory; True for contaminated DV
    mask = 'M2'
    sim = sim
    ARGS = ''
    probe = 'cosmic_shear'

    get_example_evaluate(flag_theory, probe, sim, mask, ARGS)

