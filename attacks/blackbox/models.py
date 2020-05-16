import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, backend as K, layers
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Layer, Concatenate, Dropout, \
    GlobalAveragePooling2D, Activation, GlobalMaxPooling2D, Input, AveragePooling2D, BatchNormalization
from keras_vggface import VGGFace

from attacks.blackbox import params


_weights = {'squeeze_net': params.SQUEEZENET_WEIGHTS_PATH,
            'custom': params.CUSTOM_SUB_WEIGHTS_PATH,
            'resnet50': params.RESNET50_WEIGHTS_PATH}
config = tf.ConfigProto(device_count={'GPU': 0})
graph = tf.get_default_graph()
sess = tf.Session(graph=graph, config=config)

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

    model.save_weights(weights_path, save_format='h5')

def custom_model(num_classes=params.NUM_CLASSES_VGGFACE, trained=False):
    optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
    model = tf.keras.Sequential(layers=[Conv2D(64, 2), MaxPool2D(2), Conv2D(64, 2),
                                        MaxPool2D(2), Flatten(), Dense(200, activation='sigmoid'),
                                        Dense(200, activation='sigmoid'), Dense(100, activation='relu'),
                                        Dense(num_classes, activation='softmax')])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if trained:
        model.build(input_shape=[None, 224, 224, 3])
        model.load_weights(params.CUSTOM_SUB_WEIGHTS_PATH)

    return model

def resnet50(num_classes=params.NUM_CLASSES_VGGFACE, trained=False):
    tf.keras.backend.set_session(sess)

    # optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    model = VGGFace(model='resnet50', include_top=True, weights=None, classes=num_classes)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if trained:
        model.build(input_shape=[None, 224, 224, 3])
        model.load_weights(params.RESNET50_WEIGHTS_PATH)

    return model

def squeeze_net(num_classes=params.NUM_CLASSES_VGGFACE, trained=False):
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
        model.load_weights(params.SQUEEZENET_WEIGHTS_PATH)

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
        model = VGGFace(model='resnet50')
    return model

_model_functions = {'squeeze_net': squeeze_net,
                    'custom': custom_model,
                    'resnet50': resnet50,
                    'blackbox': blackbox}
