print(f"Loading module {__file__}")

import os
os.umask(2)
LEARNING_RATE = 0.04
MOMENTUM = 0.9
EPOCHS_TRAINING = 50
EPOCHS_SUBSTITUTE = 1
BATCH_SIZE = 24
# LAMBDA = 0.01  # Step size for jacobian augmentation (for images in [0,1]
LAMBDA = 2.5  # Step size for jacobian augmentation (for images in [0,255]

PROJECT_DIR = '/cs/ep/503'  # TODO: change to os-independent path

DATASET_BASE_PATH = os.path.join(PROJECT_DIR, 'dataset')
AUGMENTATION_BASE_PATH = os.path.join(DATASET_BASE_PATH, "augmented_images")
TRAIN_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "train_aligned")
TEST_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "test_aligned")
DATASET_TRAIN_LIST = os.path.join(DATASET_BASE_PATH, "train_list.txt")
DATASET_TEST_LIST = os.path.join(DATASET_BASE_PATH, "test_list.tx")

TRAIN_SET = os.path.join(DATASET_BASE_PATH, "training_set")
VALIDATION_SET = os.path.join(DATASET_BASE_PATH, "validation_set")
TRAIN_SET_WORKING = os.path.join(DATASET_BASE_PATH, "working_dataset")
os.makedirs(TRAIN_SET_WORKING, exist_ok=True)

WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
SQUEEZENET_WEIGHTS_FNAME = 'substitute_squeezenet.h5'  # No pre-trained model at the moment
SQUEEZENET_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, SQUEEZENET_WEIGHTS_FNAME)
RESNET50_WEIGHTS_FNAME = 'substitute_resnet50.h5'
RESNET50_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, RESNET50_WEIGHTS_FNAME)
SENET50_WEIGHTS_FNAME = 'substitute_senet50.h5'
SENET50_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, SENET50_WEIGHTS_FNAME)
FACENET_WEIGHTS_FNAME = 'facenet_model.pb'
FACENET_WEIGHTS_PATH = os.path.join(PROJECT_DIR, FACENET_WEIGHTS_FNAME)
NUM_CLASSES_VGGFACE = 8631
NUM_CLASSES_RESNET50 = 195

print(f"Successfully loaded module {__file__}")