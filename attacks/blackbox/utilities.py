import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN

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

detector = MTCNN()

def extract_face(pixels, required_size=(224, 224)):
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
