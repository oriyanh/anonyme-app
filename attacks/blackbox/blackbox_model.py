import tensorflow as tf
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
from attacks.blackbox.utilities import Singleton

graph = tf.get_default_graph()
sess = tf.Session(graph=graph)


class BlackboxModel(metaclass=Singleton):
    def __init__(self):
        self.model = VGGFace(model='resnet50')

    def predict(self, input):
        self.model.predict(preprocess_input(input, version=2))


def get_blackbox_prediction(batch):
    blackbox = BlackboxModel()
    return blackbox.predict(batch)
