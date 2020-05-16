import tensorflow as tf
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
from project_utilities import Singleton

graph = tf.get_default_graph()
sess = tf.Session(graph=graph)


class BlackboxModel(metaclass=Singleton):
    """
    Singleton class representing blackbox model
    """
    def __init__(self):
        self.model = VGGFace(model='resnet50')

    def predict(self, batch):
        preprocessed_batch = preprocess_input(batch, version=2)
        preds = self.model.predict(preprocessed_batch)
        return preds


def get_blackbox_prediction(batch):
    blackbox = BlackboxModel()
    preds = blackbox.predict(batch)
    return preds
