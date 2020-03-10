import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D

from easyfacenet.simple import facenet
import attacks.blackbox.params as params


loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# class SubstituteModel(Model):
#
#     def __init__(self, num_classes):
#         self.conv1 = Conv2D(64, 2, 2)
#         self.conv2 = Conv2D(64, 2, 2)
#         self.dense1 = Dense(200, activation='relu')
#         self.dense2 = Dense(200, activation='relu')
#         self.dense3 = Dense(num_classes, activation='softmax')
#
#     def call(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return self.dense3(x)


class SubstituteModel(Model):

    def __init__(self, num_classes):
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, x):
        return self.dense(x)

def get_embeddings(images):
    return facenet.embedding(images)

def classify(model, images):
    embeddings = get_embeddings(images)
    return model(embeddings)

def load_model(weights_path, num_classes):
    model = SubstituteModel(num_classes)
    model.load_weights(weights_path)
    return model

def save_model(model, weights_path):
    model.save_weights(weights_path, save_format='h5')

def train(model, images, labels, num_epochs, batch_size):
    shuffle_seed = 10000

    train_ds = tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(shuffle_seed).batch(batch_size)

    train_step = get_train_step()
    for epoch in range(num_epochs):
        for images, labels in train_ds:
            train_step(model, images, labels)
        train_accuracy(model(images), labels)
        print(f'Epoch {epoch + 1}: Loss: {train_loss.result()}, '
              f'Accuracy: {train_accuracy.result() * 100:.2f}%')

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
    return model

def get_train_step():
    @tf.function
    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = loss_obj(labels, preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    return train_step
