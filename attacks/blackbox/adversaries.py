from attacks.blackbox.params import extract_face, SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE

from keras_vggface import utils
import tensorflow as tf
import numpy as np

CLASSIFICATION_LOSS = tf.keras.losses.CategoricalCrossentropy()
FGSM_ATTACK_NAME = 'fgsm'
FGSM_ITER_NUM = 100

def generate_adversarial_sample(image, attack, args):
    adversarial_sample = attack(image, *args)
    return adversarial_sample


def run_fgsm_attack(image, label, model, eps=0.3):
    orig_pred = np.argmax(model.predict(image))
    for i in range(FGSM_ITER_NUM):
        image = fgsm(image, label, model, eps)
        if np.argmax(model.predict(image)) != orig_pred:
            print(f"Convergence reached after {i + 1} iterations")
            break
    else:
        print("Convergence not reached")
    return image


def fgsm(x, y, model, eps=0.3, bounds=(-1., 1.)):
    """

    :param x:
    :param y:
    :param model:
    :param eps:
    :param bounds:
    :return:
    """

    (clip_min, clip_max) = bounds

    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        loss = CLASSIFICATION_LOSS(y, pred)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)

    adv_x = x + eps * signed_grad

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

def papernot():
    pass
