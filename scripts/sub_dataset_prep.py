import os
import numpy as np
import shutil
import sys
TRAIN_SIZE = 30

def dataset_prep(origin, train_dir, validation_dir, train_size):
    os.chdir(os.path.dirname(origin))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    num_classes = len(os.listdir(origin))
    for i, class_name in enumerate(os.listdir(origin)):
        class_dir = os.path.join(origin, class_name)

        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(validation_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)

        class_imgs = np.random.permutation(os.listdir(class_dir))
        train_split = min((train_size-train_size//10, len(class_imgs)))
        validation_split = min((train_split+train_size//10, len(class_imgs)))
        class_train_imgs = class_imgs[:train_split]
        class_val_imgs = class_imgs[train_split:validation_split]
        for train_img in class_train_imgs:
            shutil.copy2(os.path.join(class_dir, train_img), class_train_dir)

        for val_img in class_val_imgs:
            shutil.copy2(os.path.join(class_dir, val_img), class_val_dir)
        print(f'Divided class {i + 1}/{num_classes}: {class_name}')


if __name__ == '__main__':
    origin = sys.argv[1]
    train_dir = sys.argv[2]
    validation_dir = sys.argv[3]
    dataset_prep(origin, train_dir, validation_dir, 30)
