import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Activation, Concatenate, Dropout, Layer
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from attacks.blackbox import params


optimizer = tf.keras.optimizers.SGD(params.LEARNING_RATE, momentum=params.MOMENTUM, nesterov=True)

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

def SqueezeNet(num_classes):
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
    return model

if __name__ == '__main__':
    model = SqueezeNet(params.NUM_CLASSES_VGGFACE)
    import numpy as np


    init_b = np.random.randn(1, 224, 224, 3)
    assert model.predict(init_b) is not None
    print(f"Successfully loaded model")
    # model = SqueezeNet(weights=None, input_shape=[224, 224, 3],
    #                    classes=params.NUM_CLASSES_VGGFACE)
    # model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
