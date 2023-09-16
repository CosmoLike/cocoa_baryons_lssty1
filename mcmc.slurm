#!/bin/bash

#SBATCH --job-name=RY1_MC1
#SBATCH --output=RY1_MC1-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --account=timeifler
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pranjalrs@email.arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
source ~/.bashrc 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

echo Path is $PYTHONPATH
conda activate cocoapy38
cd /home/u31/pranjalrs/Github_repos/cocoa/Cocoa/
source start_cocoa
source ./projects/baryons_lsst_y1/scripts/start_lsst_y1
source ./projects/baryons_lsst_y1/scripts/compile_baryons_lsst_y1

echo Path is $PATH

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/baryons_lsst_y1/MCMC_Q1.yaml -r
