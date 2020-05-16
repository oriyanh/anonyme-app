import numpy as np
import tensorflow as tf
from PIL import Image
from lpips_tf import lpips


def perceptual_loss(im0_batch, im1_batch):
    """
    Evaluates Perceptual distance between input batch and output batch
    :param im0_batch: Numpy array representing batch of images
    :type im0_batch: np.ndarray
    :param im1_batch: Numpy array representing batch of images
    :type im1_batch: np.ndarray
    :return: perceptual loss between the two batches
    """
    im0_batch_ph = tf.placeholder(tf.float32, im0_batch.shape)
    im1_batch_ph = tf.placeholder(tf.float32, im1_batch.shape)

    distance_ph = lpips(im0_batch_ph, im1_batch_ph, model='net-lin', net='alex')

    with tf.get_default_session() as sess:
        dist = sess.run(distance_ph, feed_dict={
            im0_batch_ph: im0_batch,
            im1_batch_ph: im1_batch,
        })

    return dist


def extract_face(mtcnn, pixels, required_size=(224, 224),
                 graph=tf.get_default_graph()):

    # detect faces in the image
    with graph.as_default():
        results = mtcnn.detect_faces(pixels)

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


class Singleton(type):
    """
    Metaclass to be used for singleton (used for representing blackbox model
    class)
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]