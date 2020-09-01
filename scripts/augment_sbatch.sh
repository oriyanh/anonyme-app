#!/bin/csh


#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/epold/503/outputs/slurm_augment.out
#SBATCH --err=/cs/epold/503/outputs/slurm_augment.err
#SBATCH --mem=16000M
#SBATCH --time=1-0
#SBATCH --account=peleg
#SBATCH --constraint="lucy"

umask 2
setenv PYTHONPATH $PYTHONPATH\:/cs/ep/503/oriyan/repo

source /cs/epold/503/venv/bin/activate.csh
module load tensorflow
python3 /cs/epold/503/oriyan/repo/attacks/blackbox/augmentation.py


