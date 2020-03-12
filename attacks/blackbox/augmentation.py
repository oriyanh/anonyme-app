import tensorflow as tf
import numpy as np


BOUNDS = (0, 1)
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
