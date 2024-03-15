import yaml
from pathlib import Path

import cosmo_dicts

def get_example_evaluate(flag_theory, flag_create_pca, SCENARIO_NAME, ARGS, HALOFIT_VERSION, cosmo):
    template_file = yaml.safe_load(Path(f'../example_evaluate/EXAMPLE_EVALUATE_abc_template.yaml').read_text())
    
    create_baryon_pca= False
    if flag_theory is True:
        use_baryonic_simulations_for_dv_contamination = False
        which_baryonic_simulations_for_dv_contamination = ''
        prefix = 'theory'
        if flag_create_pca is True:
            create_baryon_pca= False
            

    elif flag_theory is False:
        prefix = 'baryons'
        use_baryonic_simulations_for_dv_contamination = True
        which_baryonic_simulations_for_dv_contamination = SCENARIO_NAME

    # OUTPUT
    template_file['output'] = f'./projects/baryons_lsst_y1/example_evaluate/output_log/LSST_Y1_{prefix}_{SCENARIO_NAME}{ARGS}/EXAMPLE_EVALUATE'


    # Likelihood
    likelihood = template_file['likelihood']['baryons_lsst_y1.lsst_3x2pt']

    likelihood['create_baryon_pca'] = create_baryon_pca
    likelihood['use_baryonic_simulations_for_dv_contamination'] = use_baryonic_simulations_for_dv_contamination
    likelihood['which_baryonic_simulations_for_dv_contamination'] = which_baryonic_simulations_for_dv_contamination
    likelihood['print_datavector_file'] = f'./projects/baryons_lsst_y1/data/data_vector/lsst_y1_{prefix}_{SCENARIO_NAME}{ARGS}.modelvector'

    # Theory
    extra_args = template_file['theory']['camb']['extra_args']
    extra_args['halofit_version'] = HALOFIT_VERSION


    # Sampler
    override = template_file['sampler']['evaluate']['override']
    override['As_1e9'] = cosmo['As_1e9']
    override['ns'] = cosmo['ns']
    override['H0'] = cosmo['H0']
    override['omegab'] = cosmo['omegab']
    override['omegam'] = cosmo['omegam']
    override['LSST_DZ_S1'] = 0.0
    override['LSST_DZ_S2'] = 0.0
    override['LSST_DZ_S3'] = 0.0
    override['LSST_DZ_S4'] = 0.0
    override['LSST_DZ_S5'] = 0.0
    override['LSST_M1'] = 0.0
    override['LSST_M2'] = 0.0
    override['LSST_M3'] = 0.0
    override['LSST_M4'] = 0.0
    override['LSST_M5'] = 0.0
    override['LSST_A1_1'] = 0.5
    override['LSST_A1_2'] = 0.0


    # Update template
    template_file['likelihood']['baryons_lsst_y1.lsst_3x2pt'] = likelihood
    template_file['theory']['camb']['extra_args'] = extra_args
    template_file['sampler']['evaluate']['override'] = override
    
    with open(f'{prefix}/EXAMPLE_EVALUATE_{prefix}_{SCENARIO_NAME}.yaml', 'w') as yaml_file:
        yaml.dump(template_file, yaml_file, default_flow_style=False, sort_keys=False, line_break='\t')



## Specify yaml file settings
sims = [f'Magneticum_C{i}' for i in range(1,16) if i!=8]
sims += ['Magneticum_WMAP7']

for sim in sims:
    flag_theory = False
    flag_create_pca = False
    SCENARIO_NAME = sim
    ARGS = ''
    HALOFIT_VERSION = 'takahashi'
    cosmo = getattr(cosmo_dicts, f'cosmo_{SCENARIO_NAME}')

    get_example_evaluate(flag_theory, flag_create_pca, SCENARIO_NAME, ARGS, HALOFIT_VERSION, cosmo)

