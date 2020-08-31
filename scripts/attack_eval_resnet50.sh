#!/bin/csh
#SBATCH -c2
#SBATCH --output=/cs/epold/503/outputs/eval_resnet50.out
#SBATCH --err=/cs/epold/503/outputs/eval_resnet50.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,vmem:8G
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=segalman
#SBATCH -vv


umask 2

setenv PYTHONPATH /cs/epold/503/amit
source /cs/epold/503/amit/venv/bin/activate.csh
module load tensorflow -1.14.0
module load opencv -3.4.5

# After 3 round of augmentation
python3.7 /cs/epold/503/amit/scripts/eval_blackbox_attack.py resnet50 /cs/epold/503/weights/resnet50_weights_step5/substitute_resnet50_4.h5 86 "RESNET50 substitute - LFW Benchmark" /cs/epold/503/dataset/LFW/lfw_resnet50_benchmark