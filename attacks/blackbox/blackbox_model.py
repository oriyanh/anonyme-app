import tensorflow as tf
from keras_vggface import VGGFace

graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
def get_vggface_model():
    tf.keras.backend.set_session(sess)
    model = VGGFace(model='resnet50')
    return model