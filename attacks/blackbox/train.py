import tensorflow as tf
from attacks.blackbox.substitute_model import SubstituteModel, get_embeddings
import attacks.blackbox.params as params
from attacks.blackbox.augmentation import augment_dataset


optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
loss_obj_oracle = tf.keras.losses.sparse_categorical_crossentropy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

def train(oracle, epochs_substitute, epochs_training, batch_size):
    x_train, y_train = params.load_initial_set(params.NUM_INIT_SAMPLES)

    for epoch in range(epochs_substitute):
        substitute = SubstituteModel(params.NUM_CLASSES)
        embeddings = get_embeddings(x_train)
        estimated_labels = oracle(x_train)
        train_substitute(substitute, epochs_training, embeddings, estimated_labels, batch_size)
        x_train = augment_dataset(oracle, x_train, estimated_labels)

    # TODO save weights

    return substitute

def train_substitute(substitute, num_epochs, images, labels, batch_size):
    shuffle_seed = 10000

    train_ds = tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(shuffle_seed).batch(batch_size)

    train_step = get_train_step()
    for epoch in range(num_epochs):
        for images, labels in train_ds:
            train_step(substitute, images, labels)
        train_accuracy(substitute(images), labels)
        print(f'Epoch {epoch + 1}: Loss: {train_loss.result()}, '
              f'Accuracy: {train_accuracy.result() * 100:.2f}%')

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
    return substitute

def get_train_step():
    @tf.function
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = loss_obj_oracle(labels, preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(gradients, model.trainable_variables)
        train_loss(loss)
    return train_step
