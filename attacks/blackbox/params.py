import os
from project_params import ROOT_DIR

# LEARNING_RATE = 1e-2
LEARNING_RATE = 0.04
MOMENTUM = 0.9
EPOCHS_TRAINING = 1
EPOCHS_SUBSTITUTE = 2
BATCH_SIZE = 32
LAMBDA = 0.1  # Step size for jacobian augmentation (for images in [0,1]
# LAMBDA = 25.6  # Step size for jacobian augmentation (for images in [0, 255]
NUM_INIT_SAMPLES = 1000
PROJECT_DIR = ROOT_DIR
WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'dataset', 'weights')
# WEIGHTS_DIR = r'D:\vggface2_train\weights'
SQUEEZENET_WEIGHTS_FNAME = 'substitute_squeezenet.h5'
SQUEEZENET_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, SQUEEZENET_WEIGHTS_FNAME)
CUSTOM_SUB_WEIGHTS_FNAME = 'substitute_custom.h5'
CUSTOM_SUB_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, CUSTOM_SUB_WEIGHTS_FNAME)
RESNET50_WEIGHTS_FNAME = 'substitute_resnet50.h5'
RESNET50_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, RESNET50_WEIGHTS_FNAME)
FACENET_WEIGHTS_FNAME = 'facenet_model.pb'
FACENET_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, FACENET_WEIGHTS_FNAME)
NUM_CLASSES_VGGFACE = 8631

DATASET_BASE_PATH = '/cs/ep/503/dataset'  # TODO: change to os-independent path
# DATASET_BASE_PATH = r'D:\vggface2_train'  # TODO: change to os-independent path
# DATASET_BASE_PATH = os.path.join(os.path.dirname(ROOT_DIR), 'vggface2')
DATASET_UNALIGNED_PATH = os.path.join(DATASET_BASE_PATH, "vggface2_test", "test")
DATASET_TRAIN_UNALIGNED = os.path.join(DATASET_BASE_PATH, "train")
DATASET_TEST_UNALIGNED = os.path.join(DATASET_BASE_PATH, "test")
# TRAIN_SET_ALIGNED = r"D:\vggface2_train\train_aligned"
# TRAIN_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "test_aligned")
TRAIN_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "train_aligned")
# TRAIN_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "train_aligned_subset")

TEST_SET_ALIGNED = os.path.join(DATASET_BASE_PATH, "test_aligned")
DATASET_TRAIN_LIST = os.path.join(DATASET_BASE_PATH, "train_list.txt")
DATASET_TEST_LIST = os.path.join(DATASET_BASE_PATH, "test_list.tx")

TRAIN_SET_INITIAL = os.path.join(DATASET_BASE_PATH, "initial_dataset")
TRAIN_SET_WORKING = os.path.join(DATASET_BASE_PATH, "working_dataset")