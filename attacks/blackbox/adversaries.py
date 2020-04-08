import tensorflow as tf
import numpy as np

CLASSIFICATION_LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
FGSM_ATTACK_NAME = 'fgsm'


def generate_adversarial_sample(image, model, attack, kwargs):
    adversarial_sample = attack(image[np.newaxis, ...], model, **kwargs)
    return adversarial_sample


def run_fgsm_attack(image, model, eps=0.15, num_iter=100):
    preds = model.predict(image)
    orig_pred = np.argmax(preds)
    confidence = preds[0, orig_pred]
    print(f"Initial predicted class is {orig_pred} with confidence "
          f"{confidence * 100:.03f}%")

    for i in range(num_iter):
        image = fgsm(model, tf.constant(image), tf.constant(orig_pred), eps)
        cur_preds = model.predict(image)
        pred = np.argmax(cur_preds)
        confidence = cur_preds[0, pred]
        print(f"Iteration {i + 1}/{num_iter}\n"
              f"Predicted class is {pred} with "
              f"confidence {confidence * 100:.03f}%")
        new_pred = np.argmax(model.predict(image))
        if new_pred != orig_pred:
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

    pred = model(x)
    loss = CLASSIFICATION_LOSS(y, pred)
    gradient = tf.gradients(loss, x)[0]

    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)

    adv_x = x + eps * signed_grad
    adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    adv_x = tf.math.floor(adv_x)

    # with sess.as_default():
    adv_im = tf.get_default_session().run(adv_x)

    return adv_im


def run_papernot_attack(img, model, label):
    pass


def papernot():
    pass
