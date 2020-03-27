#!/bin/csh


#SBATCH --cpus-per-task=4
#SBATCH --output=/cs/ep/503/outputs/slurm.out
#SBATCH --err=/cs/ep/503/outputs/slurm.err
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=1-0
#SBATCH --account=peleg
#SBATCH --constraint="lucy"

setenv PYTHONPATH $PYTHONPATH\:/cs/ep/503/oriyan/repo

source /cs/ep/503/venv/bin/activate.csh
module load tensorflow
python3 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py


