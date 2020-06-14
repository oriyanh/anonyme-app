import sys


print(f"Loading module {__file__}")
import os
os.umask(2)
from time import perf_counter
from datetime import timedelta
import tensorflow as tf
import numpy as np
from datetime import datetime
from PIL import Image

import attacks.blackbox.params as params
from attacks.blackbox.models import load_model
from attacks.blackbox.utilities import sess, standardize_batch


# BOUNDS = (0., 1.)
BOUNDS = (0., 255.)

with sess.as_default():
    sess.run(tf.global_variables_initializer())



def augment_dataset(model, image_dir, scale):
    """
    Augment dataset using Jacobian matrix w.r.t oracle predictions
    :param model: substitute model on which jacobian is activated.
    :param image_dir: Directory from which to iterate training images
    :param scale: Scale by which to add the signed jacobian to each image
    :return: Directory containing the augmented dataset images
    """
    batch_size = 1
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    image_it = datagen.flow_from_directory(image_dir, class_mode='sparse',
                                           batch_size=batch_size, shuffle=False,
                                           target_size=(224, 224))
    nimages = image_it.n
    nbatches = (nimages // batch_size)
    if nbatches*batch_size < nimages:
        nbatches += 1
    print("starting")
    new_image_dir = os.path.join(params.AUGMENTATION_BASE_PATH,
                                 datetime.now().strftime("%Y%m%d%H%M%S%f"), 'results')  # Nested 'results' directory is a bugfix, don't change
    print(f"Augmentation output images can be found in {new_image_dir}")
    os.makedirs(new_image_dir, mode=0o777, exist_ok=True)
    fname = 1
    save_img_fn = lambda x, y: tf.keras.preprocessing.image.save_img(os.path.join(new_image_dir, f"{y}.jpg".rjust(11, "0")), x, scale=False)


    batch_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                              name="batch_in")
    label_ph = tf.placeholder(tf.int32, shape=(),
                              name="oracle_label")
    scale_ph = tf.placeholder(tf.float32, shape=(),
                              name="oracle_label")
    diff_tensor = augment(model, batch_ph, label_ph, scale_ph)

    epoch_start_time = perf_counter()
    for nbatch in range(nbatches):
        print(f"Augmentation step # {nbatch+1}", file=sys.stdout)
        image_batch, label_batch = image_it.next()
        image_batch_standardized = standardize_batch(image_batch, True)

        with sess.as_default():
            diff= sess.run(diff_tensor, feed_dict={batch_ph: image_batch_standardized,
                                                                          label_ph: label_batch.astype(np.int)[0],
                                                                          scale_ph: scale})
        print("Saving images")
        for i, image in enumerate(image_batch):
            img = image + diff[i]
            img = np.clip(img, *BOUNDS)
            save_img_fn(img, fname)
            fname += 1

        print(f"Augmentation progress: augmented "
              f"{nbatch * batch_size + 1} / {nimages} images "
              f"({100 * (nbatch+1) / nbatches:.2f}%); ", end="", file=sys.stdout)

        time_now = perf_counter()
        time_elapsed = time_now - epoch_start_time
        time_per_step = time_elapsed / (nbatch+1)
        steps_remaining = nbatches - (nbatch+1)
        time_remaining = steps_remaining * time_per_step
        print(f"Est. time remaining: {timedelta(seconds=time_remaining)}", file=sys.stdout)
    return os.path.dirname(new_image_dir)


def augment(model, batch_tensor, label_orig, scale):
    """
    Function to augment batch of images by evaluating the substitute prediction's Jacobian matrix according
    to the oracle's predictions, and adding its sign to the original images
    :param model: Substitute model
    :param batch: Placeholder tensor containing batch of images
    :param oracle_label_batch: Placeholder tensor containing the batch's oracle predictions
    :param scale: Scale constant to add to the images
    :return:
    """

    print("a) Derivating")
    pred = model(batch_tensor)
    label_pred = pred[0, label_orig]
    print("b) calculating Jacobian")
    # Shape size is (batch_size, num_classes, 224, 224, 3)
    jacobians = tf.gradients(label_pred, batch_tensor)[0]
    # Shape size is (batch_size, 224, 224, 3)
    print("c) Augmenting")
    diff = tf.scalar_mul(
        scale, tf.sign(jacobians, name="SignJacobians"),
        name="ScaleJacobians")
    return diff


print(f"Successfully loaded module {__file__}")

if __name__ == '__main__':

    tf.keras.backend.set_session(sess)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    input_dir_it = datagen.flow_from_directory(params.TRAIN_SET_WORKING, class_mode='sparse', batch_size=4,
                                               shuffle=True, target_size=(224, 224))
    print(f"Augmenting images found in {input_dir_it.directory}")
    model = load_model('resnet50', trained=True, num_classes=input_dir_it.num_classes)
    augment_dataset(model, params.TRAIN_SET_WORKING,
                    params.LAMBDA)
    sess.close()
