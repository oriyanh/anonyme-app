import os
import numpy as np
import shutil
import sys
import attacks.blackbox.params as params

TRAIN_SPLIT = 0.8
NUM_CLASSES = 50

def dataset_prep(origin, train_dir, validation_dir, train_split):
    os.chdir(os.path.dirname(origin))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    num_classes = NUM_CLASSES
    for i, cls in enumerate(np.random.permutation(os.listdir(origin))):
        class_dir = os.path.join(origin, cls)

        train_target = os.path.join(train_dir, cls)
        validation_target = os.path.join(validation_dir, cls)
        os.makedirs(train_target, exist_ok=True)
        os.makedirs(validation_target, exist_ok=True)

        samples = np.random.permutation(os.listdir(class_dir))
        num_samples = len(samples)
        split_idx = int(num_samples * train_split)
        class_train_imgs = samples[:split_idx]
        class_val_imgs = samples[split_idx:]
        for train_sample in class_train_imgs:
            shutil.copy2(os.path.join(class_dir, train_sample), train_target)

        for val_sample in class_val_imgs:
            shutil.copy2(os.path.join(class_dir, val_sample), validation_target)
        print(f'Divided class {i + 1}/{num_classes}: {cls}')
        if (i+1) == num_classes:
            break


if __name__ == '__main__':
    os.umask(2)

    origin = sys.argv[1]
    train_dir = sys.argv[2]
    validation_dir = sys.argv[3]
    print(f"a) Dividing training set {origin} into training subset ({train_dir}) and validation subset ({validation_dir})")
    dataset_prep(origin, train_dir, validation_dir, TRAIN_SPLIT)

