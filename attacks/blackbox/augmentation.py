import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from PIL import Image
from keras_vggface import VGGFace

# from attacks.blackbox.blackbox_model import get_blackbox_prediction
import attacks.blackbox.params as params
from attacks.blackbox.models import load_model
from attacks.blackbox.utilities import sess


# BOUNDS = (0, 255)
BOUNDS = (0., 1.)

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
    batch_size = 2
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    image_it = datagen.flow_from_directory(
        image_dir, class_mode=None, batch_size=batch_size, shuffle=False,
        target_size=(224, 224))
    # files = [os.path.join(image_dir, name) for name in image_it.filenames]
    nimages = image_it.n
    nbatches = (nimages // batch_size) + 1
    new_image_dir = os.path.join(
        params.DATASET_BASE_PATH, "intermediate",
        datetime.now().strftime("%Y%m%d%H%M%S%f"), 'augmented_images')
    os.makedirs(new_image_dir, mode=0o777)
    fname = 1
    step = 0
    # load_img_fn = lambda x: np.asarray(Image.open(x)).astype(np.float32)/255.0
    save_img_fn = lambda x, y: Image.fromarray(x).save(
        os.path.join(new_image_dir, f"{y}.jpg".rjust(11, "0")))

    batch_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                              name="batch_in")
    oracle_label_batch_ph = tf.placeholder(tf.int64, shape=[None, 2],
                                           name="oracle_preds_in")

    for nbatch in range(nbatches):
        if step >= nbatches:
            break

        last_index = min(((nbatch + 1) * batch_size, nimages))
        # file_batch = files[nbatch * batch_size:last_index]
        # batch = np.asarray([load_img_fn(f) for f in file_batch])
        batch = image_it.next().astype(np.float) / 255.
        oracle_label_batch = np.argmax(model.predict(batch),
                                       axis=1)
        oracle_label_batch = np.vstack(
            [
                np.arange(len(oracle_label_batch)),
                oracle_label_batch
            ]).T

        fnames = np.arange(fname, batch.shape[0] * 2).astype(np.uint32)
        augmented_batch_tensor = augment(model, batch_ph,
                                         oracle_label_batch_ph, scale)
        with sess.as_default():
            augmented_batch = sess.run(
                augmented_batch_tensor,
                feed_dict={
                    batch_ph: batch,
                    oracle_label_batch_ph: oracle_label_batch,
                })

        for im_orig, im_augmented in zip(batch,
                                         augmented_batch):
            # save_img_fn(im_orig, fname)
            img = (im_augmented*255).astype(np.uint8)
            save_img_fn(img, fname)
            fname += 1

        fname += fnames.shape[0]
        step += 1
        print(f"Augmentation progress: augmented "
              f"{step * batch_size} / {nimages} images "
              f"({100 * step / nbatches:.2f}%)")

    return os.path.dirname(new_image_dir)


@tf.function
def augment(model, batch, oracle_label_batch, scale):
    """
    Function to augment batch of images by evaluating the substitute prediction's Jacobian matrix according
    to the oracle's predictions, and adding its sign to the original images
    :param model: Substitute model
    :param batch: Placeholder tensor containing batch of images
    :param oracle_label_batch: Placeholder tensor containing the batch's oracle predictions
    :param scale: Scale constant to add to the images
    :return:
    """

    # TODO: Change substitute architecture to support vectorization, and work without tape here
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(batch)
        label_batch = model(batch)

    # Shape size is (batch_size, num_classes, 224, 224, 3)
    jacobians = tape.batch_jacobian(label_batch, batch,
                                    experimental_use_pfor=False)

    # Shape size is (batch_size, 224, 224, 3)
    eval_jacobians = tf.gather_nd(jacobians, oracle_label_batch,
                                  name="OracePredsJacobians")
    augmented_batch = tf.scalar_mul(
        scale, tf.sign(eval_jacobians, name="SignJacobians"),
        name="ScaleJacobians")
    augmented_batch = tf.add(batch, augmented_batch,
                             name="AddScaledJacobians")
    augmented_batch = tf.clip_by_value(augmented_batch, *BOUNDS,
                                       name="Clip")
    # augmented_batch = tf.scalar_mul(augmented_batch, 255., "UpScale")
    # augmented_batch = tf.cast(augmented_batch, tf.uint8, "Cast")
    return augmented_batch

### Uncomment function in future if needed ###
# def get_augmentation_dataset(model, image_dir, scale, batch_size):
#     datagen = tf.keras.preprocessing.image.ImageDataGenerator()
#
#     image_it = datagen.flow_from_directory(image_dir, class_mode=None, batch_size=batch_size,
#                                            shuffle=False, target_size=(224, 224))
#
#     def gen():
#         while True:
#             im_batch = image_it.next()
#             im_batch_norm = utils.preprocess_input(im_batch, version=2)
#             diff = im_batch - im_batch_norm
#             im_batch_tensor = tf.constant(im_batch_norm)
#             with tf.GradientTape(persistent=True) as tape:
#                 tape.watch(im_batch_tensor)
#                 label_batch = model(im_batch_tensor)
#             jacobian = tape.gradient(label_batch, im_batch_tensor)
#             augmented_batch = im_batch_tensor + scale * tf.sign(jacobian)
#             augmented_batch += diff
#             augmented_batch = tf.clip_by_value(augmented_batch, *BOUNDS)
#             augmented_batch = tf.cast(augmented_batch, tf.uint8)
#             yield augmented_batch
#
#     ds_images = tf.data.Dataset.from_generator(gen, output_shapes=([batch_size, 224, 224, 3]),
#                                                output_types=(tf.uint8))
#     return ds_images, len(image_it.filepaths)


if __name__ == '__main__':
    # from attacks.blackbox.substitute_model import load_model

    tf.keras.backend.set_session(sess)
    # substitute_model = load_model(params.CUSTOM_SUB_WEIGHTS_PATH,
    #                               params.NUM_CLASSES_VGGFACE,
    #                               model='custom')

    # substitute_model = VGGFace(model='resnet50', weights=None)
    model = load_model('resnet50', trained=False, num_classes=25)
    augment_dataset(model, params.TRAIN_SET_WORKING,
                    params.LAMBDA)
    sess.close()
