# from attacks.blackbox.params import extract_face, SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE

from keras_vggface import utils
import tensorflow as tf
import numpy as np

CLASSIFICATION_LOSS = tf.keras.losses.CategoricalCrossentropy()
FGSM_ATTACK_NAME = 'fgsm'
FGSM_ITER_NUM = 100

def generate_adversarial_sample(image, attack, args):
    adversarial_sample = attack(image, *args)
    return adversarial_sample


def run_fgsm_attack(image, model, eps=0.3):
    preds = model.predict(image)
    orig_pred = np.argmax(model.predict(image))
    label = tf.one_hot(orig_pred, preds.shape[-1])
    label = tf.reshape(label, (1, preds.shape[-1]))
    for i in range(FGSM_ITER_NUM):
        image = fgsm(tf.constant(image), label, model, eps)
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

    return tf.Session().run(adv_x)

def papernot():
    pass


if __name__ == '__main__':
    from PIL import Image
    from attacks.blackbox.params import extract_face, SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE
    from attacks.blackbox.substitute_model import load_model
    model = load_model(SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE)
    img = extract_face(np.array(Image.open('/cs/ep/503/amit/channing_tatum.jpg'))).astype(np.float32)
    model(img[np.newaxis, ...])
    adv_img = run_fgsm_attack(img[np.newaxis, ...], model)
