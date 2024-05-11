import yaml
from pathlib import Path

import cosmo_dicts

def get_example_evaluate(flag_theory, flag_create_pca, SCENARIO_NAME, filename_arg, HALOFIT_VERSION, cosmo):
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
	template_file['output'] = f'./projects/baryons_lsst_y1/example_evaluate/output_log/LSST_Y1_{prefix}_{SCENARIO_NAME}{filename_arg}/EXAMPLE_EVALUATE'


	# Likelihood
	likelihood = template_file['likelihood']['baryons_lsst_y1.lsst_3x2pt']

	if use_EE_datavector:
		likelihood['non_linear_emul'] = 1

	else:
		likelihood['non_linear_emul'] = 2

	likelihood['create_baryon_pca'] = create_baryon_pca
	likelihood['use_baryonic_simulations_for_dv_contamination'] = use_baryonic_simulations_for_dv_contamination
	likelihood['which_baryonic_simulations_for_dv_contamination'] = which_baryonic_simulations_for_dv_contamination
	likelihood['print_datavector_file'] = f'./projects/baryons_lsst_y1/data/data_vector/lsst_y1_{prefix}_{SCENARIO_NAME}{filename_arg}.modelvector'

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
	override['LSST_B1_1'] = 1.24
	override['LSST_B1_2'] = 1.36
	override['LSST_B1_3'] = 1.47
	override['LSST_B1_4'] = 1.60
	override['LSST_B1_5'] = 1.76
	override['LSST_DZ_S1'] = 0.0
	override['LSST_DZ_S2'] = 0.0
	override['LSST_DZ_S3'] = 0.0
	override['LSST_DZ_S4'] = 0.0
	override['LSST_DZ_S5'] = 0.0
	override['LSST_DZ_L1'] = 0.0
	override['LSST_DZ_L2'] = 0.0
	override['LSST_DZ_L3'] = 0.0
	override['LSST_DZ_L4'] = 0.0
	override['LSST_DZ_L5'] = 0.0
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

	output_yaml =  f'../example_evaluate/{prefix}/EXAMPLE_EVALUATE_{prefix}_{SCENARIO_NAME}{filename_arg}.yaml'
	print(output_yaml)
	with open(output_yaml, 'w') as yaml_file:
		yaml.dump(template_file, yaml_file, default_flow_style=False, sort_keys=False, line_break='\t')



## Specify yaml file settings
FLAG_THEORY = True # False for theory; True for contaminated DV
FLAG_CREATE_PCA = False
use_EE_datavector = False
HALOFIT_VERSION = 'takahashi'



#----------------- Don't change anything below this line -----------------#
if use_EE_datavector:
	arg = '_EuclidEmu'

	# Only these cosmologies supported by EuclidEmulator2
	sims = [f'Magneticum_C{i}' for i in [7, 9, 11, 12]]
	sims += ['Magneticum_WMAP7']

else:
	if HALOFIT_VERSION == 'takahashi':
		arg = '_takahashi'

	elif HALOFIT_VERSION == 'mead2015':
		arg = '_mead2015'

	elif HALOFIT_VERSION == 'mead2020':
		arg = '_mead2020'

	else:
		raise ValueError("Invalid HALOFIT_VERSION")

	sims = [f'Magneticum_C{i}' for i in range(1, 16) if i not in [8]]
	sims += ['Magneticum_WMAP7']


for sim in sims:
	SCENARIO_NAME = sim
	cosmo = getattr(cosmo_dicts, f'cosmo_{SCENARIO_NAME}')

	get_example_evaluate(FLAG_THEORY, FLAG_CREATE_PCA, SCENARIO_NAME, arg, HALOFIT_VERSION, cosmo)

