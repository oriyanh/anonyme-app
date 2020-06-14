#!/bin/csh
#SBATCH -c2
#SBATCH --output=/cs/ep/503/outputs/resnet50_step_0.01_train.out
#SBATCH --err=/cs/ep/503/outputs/resnet50_step_0.01_train.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,vmem:8G
#SBATCH --time=4-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oriyanh
#SBATCH -vv

echo "Start train script"

umask 2

setenv PYTHONPATH /cs/ep/503/oriyan/repo
source /cs/ep/503/venv/bin/activate.csh
module load tensorflow -1.14.0

echo "Iteration 1"
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0 /cs/ep/503/dataset/training_set
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0


echo "Iteration 2"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0

echo "Iteration 3"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0

echo "Iteration 4"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0

echo "Iteration 5"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0

echo "Iteration 6"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0

echo "Iteration 7"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0

echo "Iteration 8"
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/train.py
python3.7 /cs/ep/503/oriyan/repo/attacks/blackbox/augmentation.py
python3.7 /cs/ep/503/oriyan/repo/scripts/sort_augmented_images.py 0

echo "Finished train script"
