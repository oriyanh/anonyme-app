import os
from project_params import ROOT_DIR

# LEARNING_RATE = 1e-2
LEARNING_RATE = 0.04
MOMENTUM = 0.9
EPOCHS_TRAINING = 10
EPOCHS_SUBSTITUTE = 6
BATCH_SIZE = 32
# LAMBDA = 0.1  # Step size for jacobian augmentation (for images in [0,1]
LAMBDA = 25.6  # Step size for jacobian augmentation (for images in [0, 255]
NUM_INIT_SAMPLES = 1000

WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
SQUEEZENET_WEIGHTS_FNAME = 'substitute_squeezenet.h5'
SQUEEZENET_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, SQUEEZENET_WEIGHTS_FNAME)
CUSTOM_SUB_WEIGHTS_FNAME = 'substitute_custom.h5'
CUSTOM_SUB_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, CUSTOM_SUB_WEIGHTS_FNAME)
RESNET50_WEIGHTS_FNAME = 'substitute_resnet50.h5'
RESNET50_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, RESNET50_WEIGHTS_FNAME)
FACENET_WEIGHTS_FNAME = 'facenet_model.pb'
FACENET_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, FACENET_WEIGHTS_FNAME)
NUM_CLASSES_VGGFACE = 8631

DATASET_BASE_PATH = os.path.join(ROOT_DIR, 'vggface2')
DATASET_UNALIGNED_PATH = os.path.join(DATASET_BASE_PATH, "vggface2_test", "test")
DATASET_UNALIGNED_TESTLIST = os.path.join(DATASET_BASE_PATH, "test_list.txt")
TRAINING_SET_ALIGNED_PATH = os.path.join(DATASET_BASE_PATH, "vggface2_test", "sub_training_aligned")
DATASET_ALIGNED_TRAINLIST = os.path.join(DATASET_BASE_PATH, "vggface2_test", "train_list.tx")
