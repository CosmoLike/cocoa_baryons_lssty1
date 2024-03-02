#!/bin/bash

#SBATCH --job-name=emu_train_WMAP7
#SBATCH --output=output/%x_%A.out
#SBATCH --nodes=1
# SBATCH --exclusive
#SBATCH --ntasks=90
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --time=20:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pranjalrs@email.arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
source ~/.bash_profile

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

conda activate cocoapy38
cd /home/u31/pranjalrs/Github_repos/cocoa/Cocoa/
source start_cocoa
source ./projects/baryons_lsst_y1/scripts/start_lsst_y1
#source ./projects/baryons_lsst_y1/scripts/compile_baryons_lsst_y1

CONFIGFILE=./projects/baryons_lsst_y1/emu_yaml_files/y1_cosmic_shear/magneticum_${sim}/magneticum_${sim}.yaml
export OMP_NUM_THREADS=1

if [[ $train == True ]]; then
	for N in {0..3}
	do
	    mpirun -n $SLURM_NTASKS  --mca btl ^openib,vader $PYTHON3 projects/baryons_lsst_y1/emu_cocoa_files/calculate_dv.py $CONFIGFILE $N
	    
	    echo Done calculating datavectors
	    $PYTHON3 projects/baryons_lsst_y1/emu_cocoa_files/emu_y1_cosmic_shear_train.py --config $CONFIGFILE -N $N --temper 1

	    echo Done training EMU
	    mpirun -n $SLURM_NTASKS --oversubscribe --mca pml ob1 --mca btl ^openib $PYTHON3 projects/baryons_lsst_y1/emu_cocoa_files/emu_y1_cosmic_shear_mcmc.py --config $CONFIGFILE -N $N --temper 1
	    
	    echo Done Running MCMC
	done
fi

mpirun -n $SLURM_NTASKS --mca btl tcp,self $PYTHON3 projects/baryons_lsst_y1/emu_cocoa_files/emu_y1_cosmic_shear_mcmc.py --config $CONFIGFILE -N $N --temper 0
