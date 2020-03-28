import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN


LEARNING_RATE = 1e-2
MOMENTUM = 0.9
EPOCHS_TRAINING = 10
EPOCHS_SUBSTITUTE = 6
BATCH_SIZE = 8
LAMBDA = 0.1  # Step size for jacobian augmentation
NUM_INIT_SAMPLES = 1000
VGGFACE2_BLACKBOX_WEIGHTS_PATH = '/cs/ep/503/vggface2/vggface2_Keras//model/resnet50_softmax_dim512/weights.h5'
SUBSTITUTE_WEIGHTS_PATH = '/cs/ep/503/outputs/weights/substitute_vggface.h5'
NUM_CLASSES_VGGFACE = 8631
DATASET_UNALIGNED_PATH = "/cs/ep/503/vggface2/vggface2_test/test"
DATASET_UNALIGNED_TESTLIST = '/cs/ep/503/vggface2/test_list.txt'
DATASET_ALIGNED_TRAINING = "/cs/ep/503/vggface2/vggface2_test/sub_training_aligned"
DATASET_ALIGNED_TESTLIST = "/cs/ep/503/vggface2/vggface2_test/train_list.txt"

# DATASET_ALIGNED_PATH = "/cs/ep/503/vggface2/vggface2_test/test"
detector = MTCNN()

def load_initial_set(num_samples):
    pass

def load_training_set():
    images = []
    with open(DATASET_ALIGNED_TESTLIST, 'r') as f:
        for r in f.readlines()[:1000]:
            img_path = os.path.join(DATASET_ALIGNED_TRAINING, r.strip('\n\r'))
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

if __name__ == '__main__':
    a = load_training_set()
    pass
