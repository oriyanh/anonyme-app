import tensorflow as tf


BOUNDS = (0, 1)

def augment_dataset(oracle, input_images, scale):
    predictions = oracle(input_images)
    jacobian = tf.gradients(predictions, input_images)
    augmented_images = input_images + scale * tf.sign(jacobian)
    augmented_dataset = tf.random.shuffle(tf.concat((input_images, augmented_images), axis=0))

    augmented_dataset = tf.clip_by_value(augmented_dataset, *BOUNDS)

    return augmented_dataset
