import os

import numpy as np

import attacks.blackbox.params as params
import attacks.blackbox.substitute_model as substitute
from attacks.blackbox.squeezenet import squeeze_net
from attacks.blackbox.blackbox_model import get_vggface_model, sess
from attacks.blackbox.augmentation import augment_dataset
from project_params import ROOT_DIR


def train(oracle, num_oracle_classes, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0 and nepochs_training > 0
    train_dir = params.TRAINING_SET_ALIGNED_PATH
    validation_dir = train_dir
    for epoch in range(1, nepochs_substitute+1):
        print(f"Starting training on new substitute model, epoch #{epoch}")
        # model = SubstituteModel(num_oracle_classes)
        model = squeeze_net(num_oracle_classes)
        substitute.train(model, oracle, train_dir, validation_dir, nepochs_training, batch_size)
        print("Saving")
        # substitute.save_model(model, params.SUBSTITUTE_WEIGHTS_PATH)
        substitute.save_model(model, os.path.join(
            ROOT_DIR, params.SQUEEZENET_WEIGHTS_PATH))
        if epoch < nepochs_substitute:
            print("Augmenting")
            train_dir = augment_dataset(model, train_dir, params.LAMBDA)
    return model


LOAD_WEIGHTS = False


def main():
    oracle = get_vggface_model()
    if LOAD_WEIGHTS:
        substitute_model = substitute.load_model(params.SUBSTITUTE_WEIGHTS_PATH, params.NUM_CLASSES_VGGFACE)
        init_b = np.random.randn(1, 224, 224, 3)
        assert substitute_model.predict(init_b) is not None
        print(f"Successfully loaded model")
    else:
        substitute_model = train(oracle, params.NUM_CLASSES_VGGFACE, params.EPOCHS_SUBSTITUTE,
                                 params.EPOCHS_TRAINING, params.BATCH_SIZE)


if __name__ == '__main__':
    main()
    sess.close()
