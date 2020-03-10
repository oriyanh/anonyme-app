# import os
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from keras_vggface import utils

from attacks.blackbox.blackbox_model import get_vggface_model
from attacks.blackbox.params import extract_face
from attacks.blackbox.substitute_model import SubstituteModel
import attacks.blackbox.params as params
from attacks.blackbox.augmentation import augment_dataset
from keras_vggface.vggface import VGGFace

import attacks.blackbox.substitute_model as substitute


def train(oracle, num_oracle_classes, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0
    x_train = params.load_training_set()
    x_train = np.asarray(x_train).astype(np.float)
    x_train = utils.preprocess_input(x_train, version=2)
    for epoch in range(nepochs_substitute):
        model = SubstituteModel(num_oracle_classes)
        y_train = np.argmax(oracle.predict(x_train), axis=1)
        substitute.train(model, x_train, y_train, nepochs_training, batch_size)
        pred = model(x_train)
        res = tf.argmax(pred, axis=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pred_labels = sess.run(res)
            pred_labels = pred_labels.flatten()
        res = y_train == pred_labels
        accuracy = np.count_nonzero(res) / pred_labels.shape[0]
        print(f"Accuracy after {epoch + 1} epochs: {accuracy * 100:.2f}%")
        x_train = augment_dataset(oracle, x_train, params.LAMBDA)
    substitute.save_model(model, params.SUBSTITUTE_WEIGHTS_PATH)
    return model

LOAD_WEIGHTS = False

def main():
    oracle = get_vggface_model()
    if LOAD_WEIGHTS:
        substitute_model = substitute.load_model(params.SUBSTITUTE_WEIGHTS_PATH, params.NUM_CLASSES_VGGFACE)
    else:
        substitute_model = train(oracle, params.NUM_CLASSES_VGGFACE, params.EPOCHS_SUBSTITUTE,
                                 params.EPOCHS_TRAINING, params.BATCH_SIZE)

if __name__ == '__main__':
    main()
    # model = get_vggface_model()
    #
    # img = extract_face('/cs/ep/503/oriyan/repo/channing_tatum.jpg')
    # plt.imshow(img)
    # plt.show()
    # x = img[np.newaxis, ...].astype(np.float)
    #
    # x = utils.preprocess_input(x, version=2)
    # preds = model.predict(x)
    # print(f'Predicted: {utils.decode_predictions(preds)}')
