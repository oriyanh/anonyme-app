import os
import numpy as np
from PIL import Image
from flask import current_app
from tensorflow.python.keras.backend import set_session
# from mtcnn import MTCNN

LEARNING_RATE = 1e-2
MOMENTUM = 0.9
EPOCHS_TRAINING = 10
EPOCHS_SUBSTITUTE = 6
BATCH_SIZE = 8
LAMBDA = 0.1  # Step size for jacobian augmentation
NUM_INIT_SAMPLES = 1000
WEIGHTS_DIR = os.path.join(os.path.curdir, 'weights')
VGGFACE2_BLACKBOX_WEIGHTS_PATH = '/cs/ep/503/vggface2/vggface2_Keras//model/resnet50_softmax_dim512/weights.h5'
SUBSTITUTE_WEIGHTS_FNAME = 'substitute_vggface.h5'
FACENET_WEIGHTS_FNAME = 'facenet_model.pb'
SUBSTITUTE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, SUBSTITUTE_WEIGHTS_FNAME)
FACENET_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, FACENET_WEIGHTS_FNAME)
NUM_CLASSES_VGGFACE = 8631
DATASET_UNALIGNED_PATH = "/cs/ep/503/vggface2/vggface2_test/test"
DATASET_UNALIGNED_TESTLIST = '/cs/ep/503/vggface2/test_list.txt'
DATASET_ALIGNED_TRAINING = "/cs/ep/503/vggface2/vggface2_test/sub_training_aligned"
DATASET_ALIGNED_TESTLIST = "/cs/ep/503/vggface2/vggface2_test/train_list.txt"

# DATASET_ALIGNED_PATH = "/cs/ep/503/vggface2/vggface2_test/test"
# detector = MTCNN()

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
    with current_app.graph.as_default():
        set_session(current_app.sess)
        results = current_app.mtcnn.detect_faces(pixels)

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
    from matplotlib import pyplot as plt

    a = np.array(Image.open(r'C:\Users\Segal\Desktop\Channing_Tatum_by_'
                            'Gage_Skidmore_3.jpg'))
    a = extract_face(a)
    plt.imshow(a)
    plt.show()
