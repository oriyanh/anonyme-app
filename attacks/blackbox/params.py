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

# DATASET_ALIGNED_PATH = "/cs/ep/503/vggface2/vggface2_test/test"

def load_initial_set(num_samples):
    pass

def load_test_set():
    pass
