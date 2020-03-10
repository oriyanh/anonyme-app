import tensorflow as tf


BOUNDS = (0, 1)

def augment_dataset(oracle, input_images, scale):
    images_tensor = tf.constant(input_images, dtype=tf.float32)
    predictions = oracle(images_tensor)
    jacobian = tf.gradients(predictions, images_tensor)
    augmented_images = images_tensor + scale * tf.sign(jacobian)
    augmented_dataset = tf.random.shuffle(tf.concat((input_images, augmented_images[0]), axis=0))

    augmented_dataset = tf.clip_by_value(augmented_dataset, *BOUNDS)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_dataset=sess.run(augmented_dataset)
    return new_dataset
