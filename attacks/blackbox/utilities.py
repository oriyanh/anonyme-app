import os
import numpy as np
from PIL import Image

from attacks.blackbox.params import DATASET_ALIGNED_TRAINLIST, TRAINING_SET_ALIGNED_PATH


def load_initial_set(num_samples):
    pass


def load_training_set():
    images = []
    with open(DATASET_ALIGNED_TRAINLIST, 'r') as f:
        for r in f.readlines()[:1000]:
            img_path = os.path.join(TRAINING_SET_ALIGNED_PATH, r.strip('\n\r'))
            try:
                img = Image.open(img_path)
                img_np = np.asarray(img)
                # face = extract_face(img_np)
                # images.append(face)
                images.append(img_np)
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")
    return images


def load_test_set():
    pass
