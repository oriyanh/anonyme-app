import os
from datetime import datetime
from PIL import Image
import tensorflow as tf
import numpy as np
from keras_vggface import utils

import attacks.blackbox.params as params
from keras_preprocessing.image import ImageDataGenerator


BOUNDS = (0, 255)
BATCH_SIZE = 32

def augment_dataset(model, input_images, scale):
    nbatches = (input_images.shape[0] // BATCH_SIZE) + 1
    augmented_dataset = None
    for b in range(nbatches):
        last_index = np.min(((b + 1) * BATCH_SIZE, input_images.shape[0]))
        batch = input_images[b * BATCH_SIZE:last_index]
        augmented_batch = augment_batch(model, batch, scale)
        augmented_dataset = np.concatenate((augmented_dataset, augmented_batch),
                                           axis=0) if augmented_dataset is not None else augmented_batch
        print(f"Augmentation progress: {100 * (b + 1) / nbatches:.2f}%")

    # augmented_dataset = tf.clip_by_value(augmented_dataset, *BOUNDS)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     new_dataset = sess.run(augmented_dataset)
    np.random.shuffle(augmented_dataset)
    return augmented_dataset

def augment_batch(model, batch, scale):
    images_tensor = tf.Variable(batch, dtype=tf.float64)
    predictions = model(images_tensor)
    jacobian = tf.gradients(predictions, images_tensor)
    augmented_batch = batch + scale * tf.sign(jacobian)
    augmented_batch = tf.clip_by_value(augmented_batch[0], *BOUNDS)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        augmented_batch_np = sess.run(augmented_batch)
    return augmented_batch_np

def augment_dataset2(model, image_dir, scale):
    datagen = ImageDataGenerator()
    image_it = datagen.flow_from_directory(image_dir, class_mode=None,
                                           batch_size=32,
                                           shuffle=False, target_size=(224, 224))
    nimages = len(image_it.filepaths)
    nbatches = (nimages // BATCH_SIZE) + 1
    new_image_dir = os.path.join(params.DATASET_BASE_PATH, "intermediate_images",
                                 datetime.now().strftime("%H%M%S%f"))
    os.makedirs(new_image_dir, mode=0o777)
    fname = 1
    for b in range(nbatches):
        batch = image_it.next()
        augmented_batch = augment_batch2(model, batch, scale)
        for im_array in augmented_batch:
            im = Image.fromarray(im_array)
            im.save(os.path.join(new_image_dir,
                                 f"{fname}.jpg".rjust(11, "0")))
            fname += 1
        print(f"Augmentation progress: {100 * (b + 1) / nbatches:.2f}%")

    return new_image_dir

def augment_batch2(model, batch, scale):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(batch)
        predictions = model(batch)
        pred = tf.argmax(predictions, axis=1)
    jacobian = tape.batch_jacobian(pred, batch)
    augmented_batch = batch + scale * tf.sign(jacobian)
    augmented_batch = tf.clip_by_value(augmented_batch[0], *BOUNDS)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        augmented_batch_np = sess.run(augmented_batch)
    return augmented_batch_np

def augment_dataset3(model, image_dir, scale):
    batch_size = 32
    # image_ds, nimages = get_augmentation_dataset(model, image_dir, scale, batch_size)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    image_it = datagen.flow_from_directory(image_dir, class_mode=None, batch_size=batch_size,
                                           shuffle=False, target_size=(224, 224))
    files = [os.path.join(image_dir, name) for name in image_it.filenames]
    nimages = len(image_it.filepaths)
    nbatches = (nimages // BATCH_SIZE) + 1
    new_image_dir = os.path.join(params.DATASET_BASE_PATH, "intermediate_images",
                                 datetime.now().strftime("%H%M%S%f"))
    os.makedirs(new_image_dir, mode=0o777)
    fname = 1
    step = 0
    # for step in range(nsteps):
    # for augmented_batch in image_ds:
    load_img_fn = lambda x: np.asarray(tf.keras.preprocessing.image.load_img(x, target_size=(224, 224)),
                                       dtype=np.float32)
    save_img_fn = lambda x, y: tf.keras.preprocessing.image.save_img(os.path.join(new_image_dir,
                                                                                  f"{y}.jpg".rjust(11, "0")),
                                                                     x)
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
    for nbatch in range(nbatches):
        if step >= nbatches:
            break
        last_index = min(((nbatch + 1) * batch_size, nimages))
        # batch = np.zeros((last_index-nbatch*batch_size, 224, 224, 3), dtype=np.uint8)
        file_batch = files[nbatch * batch_size:last_index]
        batch = np.asarray([load_img_fn(f) for f in file_batch])
        batch_norm = utils.preprocess_input(batch, version=2)
        diff = batch - batch_norm
        # for batch in image_it:
        # print(f"Training epoch progress: step {step + 1}/{nbatches} ({100 * (step + 1) / nbatches:.2f}%)")
        # if step >= nbatches:
        #     break
        augmented_batch = preprocess(model, batch, scale, diff)
        fnames = np.arange(fname, fname + batch.shape[0])
        fwrite = save_batch(augmented_batch, fnames)
        with sess.as_default():
            sess.run(fwrite)
        fname += batch.shape[0]
        # for im, im_augmented in zip(batch, augmented_batch):
        #     save_img_fn(im, fname)
        #     save_img_fn(im_augmented, fname+1)
        # tf.keras.preprocessing.image.save_img(os.path.join(new_image_dir,
        #                                                    f"{fname}.jpg".rjust(11, "0")), im)
        # tf.keras.preprocessing.image.save_img(os.path.join(new_image_dir,
        #                                                    f"{fname+1}.jpg".rjust(11, "0")), im_augmented)
        # im = Image.fromarray(im_array)
        # im.save(os.path.join(new_image_dir,
        #                      f"{fname}.jpg".rjust(11, "0")))
        # fname += 2
        step += 1
        print(
            f"Augmentation progress: augmented {step * batch_size * 2} / {nbatches * batch_size * 2} images ({100 * step / nbatches:.2f}%)")

    return new_image_dir

def augment_dataset4(model, image_dir, scale):
    batch_size = 32
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    image_it = datagen.flow_from_directory(image_dir, class_mode=None, batch_size=batch_size,
                                           shuffle=False, target_size=(224, 224))
    files = [os.path.join(image_dir, name) for name in image_it.filenames]
    nimages = image_it.n
    nbatches = (nimages // BATCH_SIZE) + 1
    new_image_dir = os.path.join(params.DATASET_BASE_PATH, "intermediate_images",
                                 datetime.now().strftime("%Y%m%d%H%M%S%f"))
    os.makedirs(new_image_dir, mode=0o777)
    fname = 1
    step = 0
    load_img_fn = lambda x: np.asarray(Image.open(x)).astype(np.float32)
    save_img_fn = lambda x, y: Image.fromarray(x).save(os.path.join(new_image_dir,
                                                                    f"{y}.jpg".rjust(11, "0")))
    # sess = tf.Session()
    # with sess.as_default():
    #     sess.run(tf.global_variables_initializer())
    for nbatch in range(nbatches):
        if step >= nbatches:
            break

        last_index = min(((nbatch + 1) * batch_size, nimages))
        file_batch = files[nbatch * batch_size:last_index]
        batch = np.asarray([load_img_fn(f) for f in file_batch])
        batch_norm = utils.preprocess_input(batch, version=2)
        diff = batch - batch_norm
        fnames = np.arange(fname, batch.shape[0]*2).astype(np.uint32)
        batch_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="batch_in")
        # fnames_ph = tf.placeholder(tf.uint32, shape=[None], name="fnames")
        augmented_batch_tensor = preprocess(model, batch_ph, scale, diff)
        # fwrite = save_batch(augmented_batch_tensor, fnames_ph)
        with sess.as_default():
            augmented_batch = sess.run(augmented_batch_tensor, feed_dict={batch_ph: batch_norm})
            # sess.run(fwrite, feed_dict={batch_ph: batch_norm, fnames_ph: fnames})
        for im_orig, im_augmented in zip(batch.astype(np.uint8), augmented_batch):
            save_img_fn(im_orig, fname)
            save_img_fn(im_augmented, fname + 1)
            fname += 2
        fname += fnames.shape[0]
        step += 1
        print(
            f"Augmentation progress: augmented {step * batch_size * 2} / {nbatches * batch_size * 2} images ({100 * step / nbatches:.2f}%)")
    sess.close()
    return new_image_dir

@tf.function
def preprocess(model, batch, scale, diff):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(batch)
        label_batch = model(batch)
    jacobian = tape.gradient(label_batch, batch)
    augmented_batch = tf.scalar_mul(scale, tf.sign(jacobian, name="SignJacobian"), name="ScaleJacobian")
    augmented_batch = tf.add(batch, augmented_batch, name="AddScaledJacobian")
    # augmented_batch = batch + scale * tf.sign(jacobian, name="SignJacobian")
    augmented_batch = tf.add(augmented_batch, diff, name="Decenter")
    # augmented_batch += diff
    augmented_batch = tf.clip_by_value(augmented_batch, *BOUNDS, name="Clip")
    augmented_batch = tf.cast(augmented_batch, tf.uint8, "Cast")
    return augmented_batch

# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.uint8), tf.TensorSpec(shape=None, dtype=tf.uint32)])
def save_batch(batch, fnames):
    elements = batch, fnames
    fwrite = tf.map_fn(lambda x: save_image(x[0], x[1]), elements, dtype=tf.Operation, name="WrFiles")
    return fwrite

# @tf.function
def save_image(image, fname):
    image_encoded = tf.image.encode_jpeg(image, quality=100, name="Encode")
    fname_fmt = tf.strings.format("{}.jpg", fname, name="alcFileName")
    fwrite = tf.io.write_file(fname_fmt, image_encoded, name="WrFile")
    return fwrite

def get_augmentation_dataset(model, image_dir, scale, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    image_it = datagen.flow_from_directory(image_dir, class_mode=None, batch_size=batch_size,
                                           shuffle=False, target_size=(224, 224))

    def gen():
        while True:
            # for step in range(nsteps):
            im_batch = image_it.next()
            im_batch_norm = utils.preprocess_input(im_batch, version=2)
            diff = im_batch - im_batch_norm
            # with graph.as_default():
            #     tf.keras.backend.set_session(sess)
            im_batch_tensor = tf.constant(im_batch_norm)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(im_batch_tensor)
                label_batch = model(im_batch_tensor)
                # labels = tf.argmax(label_batch, axis=1)
                # labels = tf.cast(labels, tf.float32)
            jacobian = tape.gradient(label_batch, im_batch_tensor)
            augmented_batch = im_batch_tensor + scale * tf.sign(jacobian)
            augmented_batch += diff
            augmented_batch = tf.clip_by_value(augmented_batch, *BOUNDS)
            augmented_batch = tf.cast(augmented_batch, tf.uint8)
            # print(f"Training epoch progress: {100 * (step+1) / nsteps:.2f}%")
            yield augmented_batch

    ds_images = tf.data.Dataset.from_generator(gen, output_shapes=([batch_size, 224, 224, 3]),
                                               output_types=(tf.uint8))
    return ds_images, len(image_it.filepaths)

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
if __name__ == '__main__':
    from attacks.blackbox.substitute_model import load_model

    tf.keras.backend.set_session(sess)
    substitute_model = load_model(params.SUBSTITUTE_WEIGHTS_PATH, params.NUM_CLASSES_VGGFACE)
    init_b = np.random.randn(1, 224, 224, 3)
    assert substitute_model.predict(init_b) is not None
    print(f"Successfully loaded model")
    augment_dataset4(substitute_model, params.TRAINING_SET_ALIGNED_PATH, params.LAMBDA)
    sess.close()
