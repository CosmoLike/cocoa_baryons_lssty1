import json
import numpy as np

from subprocess import call
import subprocess

sims = [f'C{i}' for i in range(1, 16) if i not in [7,8,10,11]]

sims = ['C7', 'C10', 'C11']

train = 'False'

for sim in sims:
	job_name = f'emu_train_{sim}'
	export = f'sim={sim},train={train}'
	subprocess.run(['sbatch', '--job-name '+job_name, '--export='+export, 'train_emu.sbatch'])
	print('sbatch', '--job-name '+job_name, '--export='+export, 'train_emu.sbatch')
