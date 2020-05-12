import os
import numpy as np
import tensorflow as tf
import attacks.blackbox.params as params
import attacks.blackbox.substitute_model as substitute
from attacks.blackbox import models
from attacks.blackbox.models import sess
from attacks.blackbox.augmentation import augment_dataset
from project_params import ROOT_DIR


def train(oracle, substitute_type, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0 and nepochs_training > 0
    train_dir = params.TRAIN_SET_ALIGNED
    validation_dir = train_dir
    for epoch in range(1, nepochs_substitute+1):
        print(f"Starting training on new substitute model, epoch #{epoch}")
        model = substitute.train(substitute_type, oracle, train_dir, validation_dir, nepochs_training, batch_size)
        print("Saving model")
        models.save_model(model, substitute_type)
        # if epoch < nepochs_substitute:
        #     print("Augmenting")
        #     train_dir = augment_dataset(model, train_dir, params.LAMBDA)
    return model


PRE_TRAINED = False


def main():
    oracle = models.load_model(model_type='blackbox', architecture='resnet50')
    if PRE_TRAINED:
        substitute_model = models.load_model(model_type='resnet50', trained=True)
    else:
        substitute_model = train(oracle, 'resnet50', params.EPOCHS_SUBSTITUTE,
                                 params.EPOCHS_TRAINING, params.BATCH_SIZE)


if __name__ == '__main__':
    tf.keras.backend.set_session(sess)
    main()
    sess.close()
