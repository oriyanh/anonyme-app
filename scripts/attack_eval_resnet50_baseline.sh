#!/bin/csh
#SBATCH -c2
#SBATCH --output=/cs/epold/503/outputs/eval_resnet50_baseline.out
#SBATCH --err=/cs/epold/503/outputs/eval_resnet50_baseline.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,vmem:8G
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=segalman
#SBATCH -vv


umask 2

setenv PYTHONPATH /cs/epold/503/amit
source /cs/epold/503/amit/venv/bin/activate.csh
module load tensorflow -1.14.0
module load opencv -3.4.5

# Attack baseline evaluation
python3.7 /cs/epold/503/amit/scripts/eval_blackbox_attack_baseline.py resnet50 "RESNET50 Blackbox - reg. val." /cs/epold/503/dataset/validation_set

python3.7 /cs/epold/503/amit/scripts/eval_blackbox_attack_baseline.py resnet50 "RESNET50 Blackbox - new cls. val." /cs/epold/503/dataset/validation_set_new_classes

