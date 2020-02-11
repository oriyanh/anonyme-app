import tensorflow as tf


def generate_adversarial_sample(image, attack, args):
    adversarial_sample = attack(image, *args)
    return adversarial_sample


def fgsm(x, loss, eps=0.3, bounds=(-1., 1.)):
    """

    :param x:
    :param loss:
    :param eps:
    :param bounds:
    :return:
    """

    (clip_min, clip_max) = bounds

    grad, = tf.gradients(loss, x)

    normalized_grad = tf.sign(grad)
    normalized_grad = tf.stop_gradient(normalized_grad)

    scaled_grad = eps * normalized_grad

    adv_x = x - scaled_grad

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

def papernot():
    pass
