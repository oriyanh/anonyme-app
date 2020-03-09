# import os
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from keras_vggface import utils
from mtcnn.mtcnn import MTCNN
from attacks.blackbox.substitute_model import SubstituteModel
import attacks.blackbox.params as params
from attacks.blackbox.augmentation import augment_dataset
from keras_vggface.vggface import VGGFace

import attacks.blackbox.substitute_model as substitute

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
        substitute_model = SubstituteModel()
        estimated_labels = oracle(x_train)
        substitute.train(substitute_model, x_train, estimated_labels, epochs_training, batch_size)
        x_train = augment_dataset(oracle, x_train, estimated_labels)

    # TODO save weights

    return substitute

if __name__ == '__main__':
    model = init_black_box_vggface2_keras()

    crop_size = (224, 224, 3)

    # img = extract_face('/cs/ep/503/vggface2/vggface2_test/test/n000001/0001_01.jpg')
    img = extract_face('channing_tatum.jpg')
    plt.imshow(img)
    plt.show()
    x =img[np.newaxis, ...]

    x = utils.preprocess_input(x, version=2)
    preds = model.predict(x)
    print(f'Predicted: {utils.decode_predictions(preds)}')
