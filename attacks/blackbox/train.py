# import os
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from keras_vggface import utils

import attacks.blackbox.params as params
import attacks.blackbox.substitute_model as substitute
import attacks.blackbox.utilities
from attacks.blackbox.substitute_model import SubstituteModel, SubstituteModel2
from attacks.blackbox.blackbox_model import get_vggface_model
from attacks.blackbox.augmentation import augment_dataset, augment_dataset2


def train(oracle, num_oracle_classes, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0
    x_train = attacks.blackbox.utilities.load_training_set()
    x_train = np.asarray(x_train).astype(np.float)
    x_train = utils.preprocess_input(x_train, version=2)
    for epoch in range(nepochs_substitute):
        print(f"Starting training on new substitute model, epoch #{epoch + 1}")
        model = SubstituteModel(num_oracle_classes)
        y_train = predict_in_batches(oracle, x_train, 32)
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
        x_train = augment_dataset(model, x_train, params.LAMBDA)
    substitute.save_model(model, params.SUBSTITUTE_WEIGHTS_PATH)
    return model

def train2(oracle, num_oracle_classes, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0 and nepochs_training > 0
    train_dir = params.TRAINING_SET_ALIGNED_PATH
    validation_dir = train_dir
    for epoch in range(nepochs_substitute):
        print(f"Starting training on new substitute model, epoch #{epoch + 1}")
        model = SubstituteModel2(num_oracle_classes)
        substitute.train2(model, oracle, train_dir, validation_dir, nepochs_training, batch_size)
        train_dir = augment_dataset2(model, train_dir, params.LAMBDA)
    substitute.save_model(model, params.SUBSTITUTE_WEIGHTS_PATH)
    return model

def predict_in_batches(oracle, images, batch_size):
    predictions = []
    nbatches = (images.shape[0] // batch_size) + 1
    for b in range(nbatches):
        last_index = np.min(((b + 1) * batch_size, images.shape[0]))
        batch = images[b * batch_size:last_index]
        pred = np.argmax(oracle.predict(batch), axis=1)
        predictions.extend(pred)
        print(f"Prediction progress: {100 * (b + 1) / nbatches:.2f}%")
    return np.asarray(predictions)

LOAD_WEIGHTS = False

def main():
    oracle = get_vggface_model()
    if LOAD_WEIGHTS:
        substitute_model = substitute.load_model(params.SUBSTITUTE_WEIGHTS_PATH, params.NUM_CLASSES_VGGFACE)
    else:
        # substitute_model = train(oracle, params.NUM_CLASSES_VGGFACE, params.EPOCHS_SUBSTITUTE,
        #                          params.EPOCHS_TRAINING, params.BATCH_SIZE)
        substitute_model = train2(oracle, params.NUM_CLASSES_VGGFACE, params.EPOCHS_SUBSTITUTE,
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
