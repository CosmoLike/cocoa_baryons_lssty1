import json
import numpy as np

from subprocess import call
import subprocess


probe = '3x2pt'
train = 'False'
use_hmcode = False

## Specify these if looping over these
job_choice = 2

## Option 1
if job_choice==1:
	loop_sim = True
	sims = [f'C{i}' for i in range(1, 16) if i not in [8]]
	yaml_arg = 'M1'

## Option 2
elif job_choice==2:
	loop_sim = False
	sim = 'WMAP7'
	yaml_args = ['M2', 'M3', 'M4']



## Don't change code below
hmcode_arg = '_hmcode' if use_hmcode else ''

if loop_sim:
	for sim in sims:
		job_name = f'{probe}{hmcode_args}_{sim}_{yaml_arg}'
		export = f'probe={probe},sim={sim},train={train},yaml_arg={yaml_arg},hmcode_arg={hmcode_arg}'
		subprocess.run(['sbatch', '--job-name '+job_name, '--export='+export, 'run_emu.sbatch'])
		print('sbatch', '--job-name '+job_name, '--export='+export, 'run_emu.sbatch')

## Loop over args
else:
	for yaml_arg in yaml_args:
		job_name = f'{probe}{hmcode_arg}_{sim}_{yaml_arg}'
		export = f'probe={probe},sim={sim},train={train},yaml_arg={yaml_arg},hmcode_arg={hmcode_arg}'
		subprocess.run(['sbatch', '--job-name '+job_name, '--export='+export, 'run_emu.sbatch'])
		print('sbatch', '--job-name '+job_name, '--export='+export, 'run_emu.sbatch')
