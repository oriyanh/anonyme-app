# from attacks.blackbox.params import extract_face, SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE

# from keras_vggface import utils
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

CLASSIFICATION_LOSS = tf.keras.losses.CategoricalCrossentropy()
FGSM_ATTACK_NAME = 'fgsm'


def generate_adversarial_sample(image, model, attack, kwargs):
    adversarial_sample = attack(image[np.newaxis, ...], model, **kwargs)
    return adversarial_sample


def run_fgsm_attack(image, model, eps=0.15, num_iter=100):
    preds = model.predict(image)
    orig_pred = np.argmax(preds)
    confidence = np.max(preds)
    print(f"Initial predicted class is {orig_pred} with confidence "
          f"{confidence * 100:.03f}%")

    label = tf.one_hot(orig_pred, preds.shape[-1])
    label = tf.reshape(label, (1, preds.shape[-1]))
    for i in range(num_iter):
        image = fgsm(model, tf.constant(image), label, eps)
        # image = fgsm(model, tf.constant(image), label, eps)
        # plt.imshow(np.floor(image[0]))
        # plt.show()
        # image = fgsm(image, label, model, sess, eps)
        # image = utils.preprocess_input(image[0], version=2)[np.newaxis, ...]
        cur_preds = model.predict(image)
        pred = np.argmax(cur_preds)
        confidence = np.max(cur_preds)
        print(f"Iteration {i + 1}/{num_iter}\n"
              f"Predicted class is {pred} with "
              f"confidence {confidence * 100:.03f}%")
        if np.argmax(model.predict(image)) != orig_pred:
            print(f"Convergence reached after {i + 1} iterations")
            break
    else:
        print("Convergence not reached")
    return image


def fgsm(model, x, y, eps, bounds=(0., 255.)):
    """
    Performs fgsm attack on input x with label y using prediction from given
    model.
    :param model: Model to perform attack on
    :type model: tf.Model
    :param x: Tensor of input image
    :type x: tf.Tensor
    :param y: Tensor of image label
    :type y: tf.Tensor
    :param eps: Step size for fgsm attack
    :type eps: float
    :param bounds: 2-Tuple representing image value boundaries
    :type bounds: Tuple[float, float]
    :return: Image after attack iteration
    :rtype: np.ndarray
    """

    (clip_min, clip_max) = bounds

    # with tf.GradientTape() as tape:
    #     tape.watch(x)
    #     pred = model(x)
    #     with sess.as_default():
        # loss = CLASSIFICATION_LOSS(y, pred)

    # Get the gradients of the loss w.r.t to the input image.
    # gradient = tape.gradient(loss, x)

    pred = model(x)
    loss = CLASSIFICATION_LOSS(y, pred)
    gradient = tf.gradients(loss, x)[0]

    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)

    adv_x = x + eps * signed_grad

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    adv_x = tf.math.floor(adv_x)

    # with sess.as_default():
    adv_im = tf.get_default_session().run(adv_x)

    return adv_im


def run_papernot_attack(img, model, label):
    pass


def papernot():
    pass


if __name__ == '__main__':
    from PIL import Image
    from attacks.blackbox.params import extract_face, SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE
    from attacks.blackbox.substitute_model import load_model
    from attacks.blackbox.blackbox_model import get_vggface_model
    from keras_vggface import utils

    # model = load_model(SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE)
    model = get_vggface_model()

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

    img = extract_face(np.array(Image.open('/cs/ep/503/amit/channing_tatum.jpg'))).astype(np.float32)
    img = utils.preprocess_input(img, version=2)
    # model(img[np.newaxis, ...])
    # with sess.as_default():
    adv_img = run_fgsm_attack(img[np.newaxis, ...], model, sess)
