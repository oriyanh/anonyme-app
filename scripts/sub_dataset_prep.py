import os
import numpy as np
import shutil

TEST_ROOT_DIR = '/cs/ep/503/vggface2/vggface2_test'
ORIGIN_DIR = 'test'
TRAINING_DIR = 'sub_training'
VAL_DIR = 'sub_validation'
TRAIN_SIZE = 15

def dataset_prep():
    os.chdir(TEST_ROOT_DIR)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    train_list = []
    val_list = []

    for i, class_name in enumerate(os.listdir(ORIGIN_DIR)):
        class_dir = os.path.join(ORIGIN_DIR, class_name)

        class_train_dir = os.path.join(TRAINING_DIR, class_name)
        os.makedirs(class_train_dir, exist_ok=True)

        class_val_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(class_val_dir, exist_ok=True)

        class_imgs = np.random.permutation(os.listdir(class_dir))
        class_train_imgs = class_imgs[:TRAIN_SIZE]
        class_val_imgs = class_imgs[TRAIN_SIZE:]
        for train_img in class_train_imgs:
            shutil.copy2(os.path.join(class_dir, train_img), class_train_dir)
            train_list.append(os.path.join(class_name, train_img))
        for val_img in class_val_imgs:
            shutil.copy2(os.path.join(class_dir, val_img), class_val_dir)
            val_list.append(os.path.join(class_name, val_img))

        print(f'Divided class {i + 1}/{len(os.listdir(ORIGIN_DIR))}: {class_name}')

    with open('train_list.txt', 'w') as train_file:
        train_file.writelines(train_list)

    with open('val_list.txt', 'w') as val_file:
        val_file.writelines(val_list)

if __name__ == '__main__':
    dataset_prep()
