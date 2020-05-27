#!/bin/tcsh
#SBATCH -c2
#SBATCH --output=/cs/ep/503/outputs/train.out
#SBATCH --err=/cs/ep/503/outputs/train.err
#SBATCH --mem=24576M
#SBATCH --time=2-0
#SBATCH --account=peleg
#SBATCH --constraint="oxygen"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oriyanh
#SBATCH -vv

echo "Start train script"

umask 2

source /cs/ep/503/venv/bin/activate.csh
module load tensorflow/1.14.0
setenv PYTHONPATH $PYTHONPATH\:/cs/ep/503/oriyan/repo

python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py

echo "Finished train script"
