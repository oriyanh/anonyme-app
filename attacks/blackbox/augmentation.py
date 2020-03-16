import os
from datetime import datetime
from PIL import Image
import tensorflow as tf
import numpy as np
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
        pred = tf.argmax(predictions)
    jacobian = tape.batch_jacobian(pred, batch)
    augmented_batch = batch + scale * tf.sign(jacobian)
    augmented_batch = tf.clip_by_value(augmented_batch[0], *BOUNDS)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        augmented_batch_np = sess.run(augmented_batch)
    return augmented_batch_np
