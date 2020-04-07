#!/bin/csh


#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/ep/503/outputs/slurm_train.out
#SBATCH --err=/cs/ep/503/outputs/slurm_train.err
#SBATCH --mem=16000M
#SBATCH --time=1-0
#SBATCH --account=peleg
#SBATCH --constraint="lucy"

umask 2
setenv PYTHONPATH $PYTHONPATH\:/cs/ep/503/oriyan/repo

source /cs/ep/503/venv/bin/activate.csh
module load tensorflow
python3 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
echo "Finished training, starting augmentation"
python3 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
echo "Done"


