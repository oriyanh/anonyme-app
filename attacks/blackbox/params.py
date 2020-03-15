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
