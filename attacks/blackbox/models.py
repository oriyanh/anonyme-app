print(f"Loading module {__file__}")
import os
os.umask(2)
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Layer, Concatenate, Dropout, \
    GlobalAveragePooling2D, Activation, GlobalMaxPooling2D, Input, AveragePooling2D
from keras_vggface import VGGFace, utils

from attacks.blackbox import params
from attacks.blackbox.utilities import Singleton, sess


_weights = {'squeeze_net': params.SQUEEZENET_WEIGHTS_PATH,
            'resnet50': params.RESNET50_WEIGHTS_PATH}

def load_model(model_type='squeeze_net', *args, **kwargs):
    try:
        model_fn = _model_functions[model_type]
        model = model_fn(*args, **kwargs)
    except KeyError:
        raise NotImplementedError(f"Unsupported model type: '{model_type}'")

    assert model.predict(np.random.randn(1, 224, 224, 3)) is not None

    print(f"Model '{model_type}' loaded successfully!")
    return model

def save_model(model, model_type):
    try:
        weights_path = _weights[model_type]
    except KeyError:
        raise NotImplementedError(f"Unsupported model type: '{model_type}'")

    try:
        model.save_weights(weights_path, save_format='h5')
    except TypeError:  # If model is pure keras
        model.save_weights(weights_path)

def resnet50(num_classes=params.NUM_CLASSES_VGGFACE, trained=False,
             weights_path=params.RESNET50_WEIGHTS_PATH):
    tf.keras.backend.set_session(sess)

    optimizer = keras.optimizers.Adam()
    model = VGGFace(model='resnet50', include_top=True, weights=None, classes=num_classes)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if trained:
        model.build(input_shape=[None, 224, 224, 3])
        model.load_weights(weights_path)

    return model

def squeeze_net(num_classes=params.NUM_CLASSES_VGGFACE, trained=False,
                weights_path=params.SQUEEZENET_WEIGHTS_PATH):
    optimizer = tf.keras.optimizers.SGD(params.LEARNING_RATE, momentum=params.MOMENTUM, nesterov=True)
    model = tf.keras.Sequential(layers=[Conv2D(64, 3, 2, padding='valid', activation='relu'),
                                        MaxPool2D(3, 2),
                                        FireModule(squeeze=16, expand=64),
                                        FireModule(squeeze=16, expand=64),
                                        MaxPool2D(3, 2),
                                        FireModule(squeeze=32, expand=128),
                                        FireModule(squeeze=32, expand=128),
                                        MaxPool2D(3, 2),
                                        FireModule(squeeze=48, expand=192),
                                        FireModule(squeeze=48, expand=192),
                                        FireModule(squeeze=64, expand=256),
                                        FireModule(squeeze=64, expand=256),
                                        Dropout(0.5),
                                        Conv2D(num_classes, 1, padding='valid', activation='relu'),
                                        GlobalAveragePooling2D(),
                                        Activation('softmax')])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if trained:
        model.build(input_shape=[None, 224, 224, 3])
        model.load_weights(weights_path)

    return model

class FireModule(Layer):

    def __init__(self, squeeze, expand, **kwargs):
        self.squeeze = Conv2D(squeeze, (1, 1), padding='valid', activation='relu')
        self.expand1 = Conv2D(expand, (1, 1), padding='valid', activation='relu')
        self.expand2 = Conv2D(expand, (3, 3), padding='same', activation='relu')
        self.concat = Concatenate(axis=3)
        super(FireModule, self).__init__(**kwargs)

    def call(self, x):
        x = self.squeeze(x)

        left = self.expand1(x)

        right = self.expand2(x)

        x = self.concat([left, right])
        return x

def blackbox(architecture='resnet50'):
    model = None
    tf.keras.backend.set_session(sess)
    if architecture == 'resnet50':
        model = BlackboxModel(architecture)
    return model

class BlackboxModel(metaclass=Singleton):
    """
    Singleton class representing blackbox model
    """

    def __init__(self, architecture):
        self.model = VGGFace(model=architecture)

    def predict(self, batch):
        preprocessed_batch = utils.preprocess_input(batch, version=2)
        preds = self.model.predict(preprocessed_batch)
        return preds

    def __call__(self, batch):
        return self.model(batch)


_model_functions = {'squeeze_net': squeeze_net,
                    'resnet50': resnet50,
                    'blackbox': blackbox}

print(f"Successfully loaded module {__file__}")
