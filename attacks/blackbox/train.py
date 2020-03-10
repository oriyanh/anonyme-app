# import os
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
from keras_vggface import utils
from mtcnn.mtcnn import MTCNN
from attacks.blackbox.substitute_model import SubstituteModel
import attacks.blackbox.params as params
from attacks.blackbox.augmentation import augment_dataset
from keras_vggface.vggface import VGGFace

import attacks.blackbox.substitute_model as substitute

LOAD_WEIGHTS = False


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


def train(oracle, num_oracle_classes, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0
    x_train, y_train = params.load_initial_set(params.NUM_INIT_SAMPLES)
    for epoch in range(nepochs_substitute):
        substitute_model = SubstituteModel(num_oracle_classes)
        estimated_labels = oracle.predict(x_train)
        substitute.train(substitute_model, x_train, estimated_labels, nepochs_training, batch_size)
        x_train = augment_dataset(oracle, x_train, estimated_labels)

    substitute.save_model(substitute_model, params.SUBSTITUTE_WEIGHTS_PATH)
    return substitute_model


def main():
    oracle = VGGFace(model='resnet50')
    if LOAD_WEIGHTS:
        substitute_model = substitute.load_model(params.SUBSTITUTE_WEIGHTS_PATH, params.NUM_CLASSES_VGGFACE)
    else:
        substitute_model = train(oracle, params.NUM_CLASSES_VGGFACE,params.EPOCHS_SUBSTITUTE, params.EPOCHS_TRAINING, params.BATCH_SIZE)


if __name__ == '__main__':

    # load image from file
    model = VGGFace(model='resnet50')
    img = plt.imread('../../channing_tatum.jpg')
    preds = run_blackbox(img, model)
    print(f'Predicted: {utils.decode_predictions(preds)}')
