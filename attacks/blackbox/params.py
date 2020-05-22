print(f"Loading module {__file__}")

import os

LEARNING_RATE = 0.04
MOMENTUM = 0.9
EPOCHS_TRAINING = 10
EPOCHS_SUBSTITUTE = 5
BATCH_SIZE = 64
LAMBDA = 0.1  # Step size for jacobian augmentation (for images in [0,1]
NUM_INIT_SAMPLES = 1000

PROJECT_DIR = '/cs/ep/503'  # TODO: change to os-independent path

DATASET_BASE_PATH = os.path.join(PROJECT_DIR, 'dataset')
TRAIN_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "train_aligned")
TEST_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "test_aligned")
DATASET_TRAIN_LIST = os.path.join(DATASET_BASE_PATH, "train_list.txt")
DATASET_TEST_LIST = os.path.join(DATASET_BASE_PATH, "test_list.tx")

TRAIN_SET = os.path.join(DATASET_BASE_PATH, "train_aligned_subset")
VALIDATION_SET = os.path.join(DATASET_BASE_PATH, "validation_aligned")
TRAIN_SET_WORKING = os.path.join(DATASET_BASE_PATH, "working_dataset")
if not os.path.exists(TRAIN_SET_WORKING):
    os.makedirs(TRAIN_SET_WORKING)

WEIGHTS_DIR = os.path.join(DATASET_BASE_PATH, 'weights')
SQUEEZENET_WEIGHTS_FNAME = 'substitute_squeezenet.h5'  # No pre-trained model at the moment
SQUEEZENET_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, SQUEEZENET_WEIGHTS_FNAME)
RESNET50_WEIGHTS_FNAME = 'substitute_resnet50.h5'
RESNET50_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, RESNET50_WEIGHTS_FNAME)
FACENET_WEIGHTS_FNAME = 'facenet_model.pb'
FACENET_WEIGHTS_PATH = os.path.join(PROJECT_DIR, FACENET_WEIGHTS_FNAME)
NUM_CLASSES_VGGFACE = 8631
NUM_CLASSES_RESNET50 = 195

print(f"Successfully loaded module {__file__}")