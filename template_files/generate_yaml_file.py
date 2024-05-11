import os
from pathlib import Path
import yaml

import sys
sys.path.append('../../example_evaluate')

import cosmo_dicts

def get_example_evaluate(flag_theory, probe, sim, mask, ARGS, cosmo_params, non_lin_model='Takahashi', update_baryons=False, isMagneticum=True):
    
    template_file = yaml.safe_load(Path(f'{probe}_template.yaml').read_text())

    if flag_theory:
        prefix = '_theory'
        template_file['emulator']['baryons']['n_pcas_baryon'] = 0

    else:
        prefix = ''

    # Likelihood
    template_file['likelihood'][f'baryons_lsst_y1.lsst_{probe}']['data_file'] = f'LSST_Y1_emu_{probe}_{sim}_{mask}{prefix}.dataset'  #over-written for hmcode later
    
    template_file['params']['omegabh2']['prior']['loc'] = cosmo_params['omegab']*(cosmo_params['H0']/100)**2
    
    template_file['params']['As_1e9']['ref']['loc'] = cosmo_params['As_1e9']
    template_file['params']['omegabh2']['ref']['loc'] = cosmo_params['omegab']*(cosmo_params['H0']/100)**2
    template_file['params']['omegam']['ref']['loc'] = cosmo_params['omegam']
    template_file['params']['H0']['ref']['loc'] = cosmo_params['H0']

    # In case we want to use HMcode instead of PCA
    if non_lin_model!='Takahashi':
        # Also set npcas to zero
        template_file['emulator']['baryons']['n_pcas_baryon'] = 0
        template_file['emulator']['baryons'].pop('prior_Q1', None)
        template_file['emulator']['baryons'].pop('prior_Q2', None)
        template_file['emulator']['baryons'].pop('prior_Q3', None)

        template_file['likelihood'][f'baryons_lsst_y1.lsst_{probe}']['data_file'] = f'LSST_Y1_emu_{probe}_{sim}_{mask}_{non_lin_model}.dataset'

        if non_lin_model == 'hmcode':
            probe = probe + '_hmcode'
            TAGN_dict = {'prior': {'min': 7, 'max': 9},
			      'ref': {'dist': 'norm', 'loc': 7.2, 'scale': 0.3},
			      'proposal': 0.1, 'latex': '\log \mathrm{T}_{\mathrm{AGN}}'}
		
            # Insert before w0pwa
            pos = list(template_file['params'].keys()).index('w0pwa')
            items = list(template_file['params'].items())
            items.insert(pos, ('HMCode_logT_AGN', TAGN_dict))
            template_file['params'] = dict(items)

            extra_args = template_file['theory']['camb']['extra_args']
            extra_args['halofit_version'] = 'mead2020_feedback'
        
        elif non_lin_model == 'mead2015':
            probe = probe + '_mead2015'
            A_baryon_dict= {'prior': {'min': 0.5, 'max': 10},
	   		      'ref': {'dist': 'norm', 'loc': 3.2, 'scale': 0.3},
	   		      'proposal': 0.1, 'latex': 'A_{\mathrm{baryon}}'}
	   	
            eta_baryon_dict= {'prior': {'min': 0.1, 'max': 1.2},
	   		      'ref': {'dist': 'norm', 'loc': 0.603, 'scale': 0.1},
	   		      'proposal': 0.1, 'latex': '\eta_{\mathrm{baryon}}'}

	    # Insert before w0pwa
            pos = list(template_file['params'].keys()).index('w0pwa')
            items = list(template_file['params'].items())
            items.insert(pos, ('HMCode_A_baryon', A_baryon_dict))

            pos +=1
            items.insert(pos, ('HMCode_eta_baryon', eta_baryon_dict))
            template_file['params'] = dict(items)

            extra_args = template_file['theory']['camb']['extra_args']
            extra_args['halofit_version'] = 'mead2015'
	
        else:
            raise Exception(f'Uknown non-linear model {non_lin_model}!')
        
    # Emulator
    emulator = template_file['emulator']
    proj_dir_path = './projects/baryons_lsst_y1/'
    savedir =  f'y1_{probe}/{sim}/emu_output' # Used for saving chains

    if isMagneticum:
	    emudir = f'/xdisk/timeifler/pranjalrs/baryons_lsst_y1_emu/y1_{probe}/{sim}/' # Used for saving training samples and emu
	    emulator['training']['dv_fid'] = proj_dir_path + f'data/data_vector/lsst_y1_theory_{sim}.modelvector'
    else:
	    emudir = f'/xdisk/timeifler/pranjalrs/baryons_lsst_y1_emu/y1_{probe}/Magneticum_WMAP7/' # Used for saving training samples and emu
	    emulator['training']['dv_fid'] = proj_dir_path + f'data/data_vector/lsst_y1_theory_Magneticum_WMAP7.modelvector'

    if not os.path.isdir(savedir):
        print(f'Creating {savedir}...')
        os.makedirs(savedir)

    if not os.path.isdir(emudir):
        print(f'Creating {emudir}...')
        os.makedirs(emudir)


    emulator['io']['savedir'] = proj_dir_path + 'emu/yaml_files/' + savedir
    emulator['io']['emudir'] = emudir
    emulator['io']['chainname'] = f'chain{prefix}_{mask}{ARGS}'


    ## HACK: useful when use_hmcode is True and `probe` include '_hmcode'
    if 'cosmic_shear' in probe:
	    emulator['sampling']['scalecut_mask'] = proj_dir_path + f'data/masks/LSST_Y1_cosmic_shear_{mask}.mask'

    elif '3x2pt' in probe:
	    emulator['sampling']['scalecut_mask'] = proj_dir_path + f'data/masks/LSST_Y1_3x2pt_{mask}.mask'

    # Update PC prior
    if update_baryons:
        assert ARGS!=''
        print('Using informative PC prior to (-5.,25) (-2, 4) (-2.5, 2)...')
        emulator['baryons']['prior_Q1'] = '-5.,25.'
        emulator['baryons']['prior_Q2'] = '-2.,4.'
        emulator['baryons']['prior_Q3'] = '-2.5,2.'

    if not os.path.isdir(f'y1_{probe}/{sim}/'):
        print(f'Creating y1_{probe}/{sim}...')
        os.makedirs(f'y1_{probe}/{sim}/')
    

    file_name = f'y1_{probe}/{sim}/{sim}{prefix}_{mask}{ARGS}.yaml'
    print(file_name)
    print('\n')
    with open(file_name, 'w') as yaml_file:
        yaml.dump(template_file, yaml_file, default_flow_style=False, sort_keys=False, line_break='\t')


## Specify yaml file settings
#sims = [f'Magneticum_C{i}' for i in range(1,16) if i!=8]
sims = ['Magneticum_WMAP7']

for sim in sims:
    flag_theory = False # True for theory; False for contaminated DV
    mask = 'M2'
    sim = sim
    probe = 'cosmic_shear'
    update_baryons = True # in case we need to change prior on Q1,Q2,Q3
    ARGS = '_informPCprior' # Always start with `_`
    non_lin_model = 'Takahashi' 

    try:
	    cosmo = getattr(cosmo_dicts, f'cosmo_{sim}')
	    isMagneticum = True # Set automatically
    except AttributeError:
            isMagneticum = False
            print('Defaulting to WMAP7 cosmology')
            cosmo = getattr(cosmo_dicts, f'cosmo_Magneticum_WMAP7')

    get_example_evaluate(flag_theory, probe, sim, mask, ARGS, cosmo, non_lin_model, update_baryons, isMagneticum)