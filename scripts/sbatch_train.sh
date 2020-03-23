#!/bin/csh


#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/ep/503/outputs/slurm
#SBATCH --mem-per-cpu=3000M
#SBATCH --account=peleg

setenv PYTHONPATH $PYTHONPATH\:/cs/ep/503/oriyan/repo

source /cs/ep/503/venv/bin/activate.csh
module load tensorflow
python3 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py


