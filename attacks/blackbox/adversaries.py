from attacks.blackbox.utilities import standardize_batch

print(f"Loading module {__file__}")

import tensorflow as tf
import numpy as np

CLASSIFICATION_LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
FGSM_ATTACK_NAME = 'fgsm'


def generate_adversarial_sample(model, im_batch, attack, kwargs, sess=tf.get_default_session()):
    """
    Generates adversarial samples for given image batch using given attack and arguments
    :param model: Model to attack
    :param im_batch: Batch of images from which to creates adversarial samples
    :param attack: callable representing adversarial attack
    :param kwargs: attack-specific keyword arguments
    :param sess: tensorflow session on which to perform the attack
    :return: adversarial samples generated from im_batch
    """
    # im_batch = im_batch.astype(np.float32) / 255.0
    adv_im_batch = attack(model, im_batch, sess, **kwargs)
    # adv_im_batch = (adv_im_batch * 255.0).astype(np.uint8)
    return adv_im_batch


def calc_preds(model, im_batch):
    """
    Calculates class and confidence of predictions
    :param model:
    :param im_batch:
    :return:
    """
    batch_preds = model.predict(im_batch)
    preds = np.argmax(batch_preds, axis=-1)
    confidence = batch_preds[np.arange(len(preds)), preds]

    # print(f"Confidence: \n{np.column_stack((preds, confidence))}")
    return preds, confidence


def run_fgsm_attack(model, im_batch, sess=tf.get_default_session(), eps=0.05, num_iter=100, to_convergence=True,
                    preprocess_func=standardize_batch, attack_bounds=(0., 255.)):
    res_batch = np.copy(im_batch)
    adv_diff_batch = np.zeros_like(res_batch)
    batch_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                              name="batch_in")
    label_ph = tf.placeholder(tf.int32, shape=[None], name="pred_in")
    adv_im_batch = fgsm(model, batch_ph, label_ph, eps)

    orig_preds, _ = calc_preds(model, res_batch)
    preds = orig_preds
    idx_mask = np.array([True] * len(preds))

    for i in range(num_iter):
        print(f"Iteration {i + 1}/{num_iter}")
        adv_diff_batch[idx_mask] = sess.run(
            adv_im_batch,
            feed_dict={
                batch_ph: preprocess_func(res_batch[idx_mask]) if preprocess_func else res_batch[idx_mask],
                label_ph: orig_preds[idx_mask],
            })
        res_batch[idx_mask] += adv_diff_batch[idx_mask]
        res_batch = np.clip(res_batch, *attack_bounds)
        preds, _ = calc_preds(model, res_batch)

        # Perform adversarial attack on images yet to converge
        idx_mask = preds == orig_preds if to_convergence else idx_mask

        if not np.any(idx_mask):
            print(f"Convergence reached after {i + 1} iterations")
            break
    else:
        if to_convergence:
            print("Convergence not reached")
    return res_batch


def fgsm(model, x, y, eps):
    """
    Performs fgsm attack on input x with label y using prediction from given
    model.
    :param model: Model to perform attack on
    :type model: tf.Model
    :param x: Tensor of input image
    :type x: tf.Placeholder
    :param y: Tensor of image label
    :type y: tf.Placeholder
    :param eps: Step size for fgsm attack
    :type eps: float
    :return: Image after attack iteration
    """

    pred = model(x)

    loss = CLASSIFICATION_LOSS(y, pred)
    gradient = tf.gradients(loss, x)[0]

    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)

    return eps * signed_grad


def run_papernot_attack(model, img, sess):
    pass


def papernot():
    pass


print(f"Successfully loaded module {__file__}")
