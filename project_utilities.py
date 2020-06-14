from PIL import Image
import numpy as np
import tensorflow as tf
import torch


def perceptual_loss(lpips_network, im0_batch, im1_batch):
    """
    Evaluates Perceptual distance between input batch and output batch
    :param im0_batch: Numpy array representing batch of images
    :type im0_batch: np.ndarray
    :param im1_batch: Numpy array representing batch of images
    :type im1_batch: np.ndarray
    :return: perceptual loss between the two batches
    """

    use_cuda = torch.cuda.is_available()

    ch_first_im0_batch = np.rollaxis(im0_batch, -1, 1)
    im0_batch_pyt = torch.as_tensor((ch_first_im0_batch / (255. / 2.)) - 1)
    if use_cuda:
        im0_batch_pyt = im0_batch_pyt.cuda()

    ch_first_im1_batch = np.rollaxis(im1_batch, -1, 1)
    im1_batch_pyt = torch.as_tensor((ch_first_im1_batch / (255. / 2.)) - 1)
    if use_cuda:
        im1_batch_pyt = im1_batch_pyt.cuda()

    dist_tensor = lpips_network.forward(im0_batch_pyt, im1_batch_pyt)
    if use_cuda:
        dist_tensor = dist_tensor.cpu()
    dist = dist_tensor.data.numpy()

    return dist.reshape(im1_batch.shape[0])


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