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

GPU_CONST = '2'


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
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
    return face_array.astype('float32')

def init_black_box_vggface2_keras():
    return VGGFace(model='resnet50')



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
    model = init_black_box_vggface2_keras()

    crop_size = (224, 224, 3)
    # img = PIL.Image.open('/cs/ep/503/vggface2/vggface2_test/test/n000001/0001_01.jpg')
    # im_shape = np.array(img.size)
    # img = img.convert('RGB')

    # img = extract_face('/cs/ep/503/vggface2/vggface2_test/test/n000001/0001_01.jpg')
    img = extract_face('channing_tatum.jpg')
    plt.imshow(img)
    plt.show()
    x = np.expand_dims(img, axis=0)

    x = utils.preprocess_input(x, version=2)
    preds = model.predict(x)
    print(f'Predicted: {utils.decode_predictions(preds)}')


    # ratio = float(224) / np.min(im_shape)
    # img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
    #                        int(np.ceil(im_shape[1] * ratio))),  # height
    #                  resample=PIL.Image.BILINEAR)
    # img = np.array(img)
    # newshape = img.shape[:2]
    # h_start = (newshape[0] - crop_size[0])//2
    # w_start = (newshape[1] - crop_size[1])//2
    # img = img[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    # img = img[:, :, ::-1] - (91.4953, 103.8827, 131.0912)
    # prediction = model.predict(img.reshape(*((-1, ) + img.shape)), batch_size=1)
    # print(prediction)
