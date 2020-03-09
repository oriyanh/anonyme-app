# import os
from PIL import Image
from keras.preprocessing import image
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
# from attacks.blackbox.vgg_blackbox import Vggface2_ResNet50
from keras_vggface import utils
from mtcnn.mtcnn import MTCNN
from attacks.blackbox.substitute_model import SubstituteModel
import attacks.blackbox.params as params
from attacks.blackbox.augmentation import augment_dataset
from keras_vggface.vggface import VGGFace

optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
loss_obj_oracle = tf.keras.losses.sparse_categorical_crossentropy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


def extract_face(pixels, required_size=(224, 224)):
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# def run_blackbox(img, model):
def run_blackbox(img):
    model = VGGFace(model='resnet50')
    img = extract_face(img)
    x = img[np.newaxis, ...].astype(np.float)
    x = utils.preprocess_input(x, version=2)
    return model.predict(x)
    # print(f'Predicted: {utils.decode_predictions(preds)}')


def train(oracle, epochs_substitute, epochs_training, batch_size):
    x_train, y_train = params.load_initial_set(params.NUM_INIT_SAMPLES)

    for epoch in range(epochs_substitute):
        substitute = SubstituteModel()
        estimated_labels = oracle(x_train)
        train_substitute(substitute, epochs_training, x_train, estimated_labels, batch_size)
        x_train = augment_dataset(oracle, x_train, estimated_labels)

    # TODO save weights

    return substitute


def train_substitute(substitute, num_epochs, images, labels, batch_size):
    shuffle_seed = 10000

    train_ds = tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(shuffle_seed).batch(batch_size)

    train_step = get_train_step()
    for epoch in range(num_epochs):
        for images, labels in train_ds:
            train_step(substitute, images, labels)
        train_accuracy(substitute(images), labels)
        print(f'Epoch {epoch + 1}: Loss: {train_loss.result()}, '
              f'Accuracy: {train_accuracy.result() * 100:.2f}%')

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
    return substitute

def get_train_step():
    @tf.function
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = loss_obj_oracle(labels, preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(gradients, model.trainable_variables)

    return train_step


if __name__ == '__main__':

    # load image from file
    model = VGGFace(model='resnet50')
    img = plt.imread('../../channing_tatum.jpg')
    preds = run_blackbox(img, model)
    print(f'Predicted: {utils.decode_predictions(preds)}')
