import os
import attacks.blackbox.params as params
from attacks.blackbox import models
from attacks.blackbox.utilities import oracle_classify_and_save
import sys
os.umask(2)

if __name__ == '__main__':
    threshold = 0.5 if len(sys.argv) <= 1 else float(sys.argv[1])
    dst = params.TRAIN_SET_WORKING if len(sys.argv) <= 3 else sys.argv[3]
    if len(sys.argv) > 2:
        src = sys.argv[2]
    else:
        augmented_images_dir = params.AUGMENTATION_BASE_PATH
        dirs = os.listdir(augmented_images_dir)
        dirs_sorted = sorted(dirs, reverse=True)
        latest_dir = dirs_sorted[0]
        src = os.path.join(augmented_images_dir, latest_dir)

    os.makedirs(dst, exist_ok=True)
    oracle = models.load_model(model_type='blackbox', architecture='resnet50')
    print(f"Predicting dataset {src} and sorting into directory {dst}")
    oracle_classify_and_save(oracle, src, dst, params.BATCH_SIZE, prune_threshold=threshold, min_num_samples=2)
