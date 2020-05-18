#!/bin/csh


#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/ep/503/outputs/train.out
#SBATCH --err=/cs/ep/503/outputs/train.err
#SBATCH --mem=32000M
#SBATCH --time=2-0
#SBATCH --account=peleg
#SBATCH --constraint="lucy"

umask 2

source /cs/ep/503/venv/bin/activate.csh
module load tensorflow/1.14.0
setenv PYTHONPATH $PYTHONPATH\:/cs/ep/503/oriyan/repo
echo "Start train script"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
#echo "Finished training, starting augmentation"
#python3 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
echo "Finished train script"
