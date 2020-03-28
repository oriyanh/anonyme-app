import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input

from easyfacenet.simple import facenet
import attacks.blackbox.params as params
import numpy as np

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

class SubstituteModel(Model):

    def __init__(self, num_classes):
        super(SubstituteModel, self).__init__()
        self.conv1 = Conv2D(64, 2)
        self.maxpool1 = MaxPool2D(2)
        self.conv2 = Conv2D(64, 2)
        self.maxpool2 = MaxPool2D(2)
        self.flatten = Flatten()
        self.dense1 = Dense(200, activation='sigmoid')
        self.dense2 = Dense(200, activation='sigmoid')
        self.dense2 = Dense(100, activation='relu')
        self.dense3 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


def SubstituteModel2(num_classes):
    model = tf.keras.Sequential(layers=[Conv2D(64, 2), MaxPool2D(2), Conv2D(64, 2),
                                        MaxPool2D(2), Flatten(), Dense(200, activation='sigmoid'),
                                        Dense(200, activation='sigmoid'), Dense(100, activation='relu'),
                                        Dense(num_classes, activation='softmax')])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_embeddings(images):
    return facenet.embedding(images)


def classify(model, images):
    embeddings = get_embeddings(images)
    return model(embeddings)

def load_model(weights_path, num_classes, input_shape=(224, 224, 3)):
    input_tensor = Input(input_shape)
    model = SubstituteModel2(num_classes)
    model(input_tensor)
    model.load_weights(weights_path)
    # model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

def save_model(model, weights_path):
    model.save_weights(weights_path, save_format='h5')


def get_batches(images, labels, batch_size):
    nbatches = (images.shape[0] // batch_size) + 1
    permutation = np.random.permutation(np.arange(images.shape[0]))
    images_perm = images[permutation]
    labels_perm = labels[permutation]
    for b in range(nbatches):
        last_index = np.min(((b + 1) * batch_size, images.shape[0]))
        x_train = images_perm[b * batch_size:last_index]
        y_train = labels_perm[b * batch_size:last_index]
        yield x_train, y_train

def train(model, images, labels, num_epochs, batch_size):
    # shuffle_seed = 1000

    # train_ds = tf.data.Dataset.from_tensor_slices(
    #     (images, labels)).shuffle(shuffle_seed).batch(batch_size)

    train_step = get_train_step()
    nbatches = images.shape[0] // batch_size
    for epoch in range(num_epochs):
        print(f"Start training epoch #{epoch+1}")
        nbatch = 0
        for im_batch, label_batch in get_batches(images, labels, batch_size):
            nbatch += 1
            if nbatch > nbatches:
                break
            print(f"Training epoch progress: {100*nbatch/nbatches:.2f}%")
            train_step(model, im_batch, label_batch)
        # train_accuracy(labels, model(images))
        # with tf.Session().as_default():
        #     print(f'Epoch {epoch + 1}: Loss: {train_loss.result().eval()}, '
        #           f'Accuracy: {train_accuracy.result().eval() * 100:.2f}%')

        # Reset the metrics for the next epoch
        # train_loss.reset_states()
        # train_accuracy.reset_states()
    # pred = model(images)
    # res = tf.argmax(pred, axis=1)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     pred_labels=sess.run(res)
    #     pred_labels = pred_labels.flatten()
    # res = labels == pred_labels
    # accuracy = np.count_nonzero(res) / pred_labels.shape[0]
    # print(f"Accuracy after {num_epochs} epochs: {accuracy * 100:.2f}%")
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
