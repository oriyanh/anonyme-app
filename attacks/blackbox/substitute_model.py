import tensorflow as tf
from keras_vggface import utils
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from easyfacenet.simple import facenet
import attacks.blackbox.params as params
import numpy as np
from attacks.blackbox.blackbox_model import graph, sess

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

def load_model(weights_path, num_classes):
    model = SubstituteModel(num_classes)
    model.load_weights(weights_path)
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
        print(f"Start training epoch #{epoch + 1}")
        nbatch = 0
        for im_batch, label_batch in get_batches(images, labels, batch_size):
            nbatch += 1
            if nbatch > nbatches:
                break
            print(f"Training epoch progress: {100 * nbatch / nbatches:.2f}%")
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

def train2(model, oracle, train_dir, validation_dir, num_epochs, batch_size):
    train_gen, nsamples = training_generator(oracle, train_dir, batch_size)
    train_ds, nsamples = get_training_set(oracle, train_dir, batch_size)
    train_step = get_train_step()
    nsteps = (nsamples // batch_size) + 1
    # model.fit_generator(iter(train_ds), epochs=num_epochs, steps_per_epoch=2, verbose=1)
    # model.fit(train_ds, epochs=num_epochs, steps_per_epoch=nsteps, verbose=1)
    for epoch in range(num_epochs):
        print(f"Start training epoch #{epoch + 1}")
        step = 0
        # for step in range(nsteps):
        for im_batch, label_batch in train_ds:
            print(f"Training epoch progress: step {step+1}/{nsteps} ({100 * (step+1) / nsteps:.2f}%)")
            if step >= nsteps:
                    break
    #         model.train_on_batch(train_ds)
            train_step(model, im_batch, label_batch)
            step += 1

    val_gen = validation_generator(oracle, validation_dir, batch_size)
    # [loss, accuracy] = model.evaluate(train_ds, steps=2)
    # print(f"Total loss on validation set: {loss:.2f} ; Accuracy: {accuracy:.2f}")
    return model

def training_generator(oracle, train_dir, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_it = datagen.flow_from_directory(train_dir, class_mode=None,
                                           batch_size=batch_size,
                                           shuffle=True, target_size=(224, 224))
    nsteps = (len(train_it.filepaths) // batch_size) + 1
    def gen():
        while True:
            # for step in range(nsteps):
            im_batch = train_it.next()
            x_train = utils.preprocess_input(im_batch, version=2)
            with graph.as_default():
                tf.keras.backend.set_session(sess)
                label_batch = oracle.predict(x_train)
            y_train = np.argmax(label_batch, axis=1)
            yield x_train, y_train
        # print(f"Training epoch progress: {100 * step / nsteps:.2f}%")
    return gen, nsteps
def get_training_set(oracle, train_dir, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_it = datagen.flow_from_directory(train_dir, class_mode=None, batch_size=batch_size,
                                           shuffle=True, target_size=(224, 224))
    def gen():

        while True:
            # for step in range(nsteps):
            im_batch = train_it.next()
            x_train = utils.preprocess_input(im_batch, version=2)
            with graph.as_default():
                tf.keras.backend.set_session(sess)
                label_batch = oracle.predict(x_train)
            y_train = np.argmax(label_batch, axis=1)
            # print(f"Training epoch progress: {100 * (step+1) / nsteps:.2f}%")
            yield x_train, y_train
    ds_images = tf.data.Dataset.from_generator(gen, output_shapes=([batch_size, 224, 224, 3], [batch_size]),
                                               output_types=(tf.float32, tf.int32))
    return ds_images, len(train_it.filepaths)
    # def preprocess(image):
    #     x_train = utils.preprocess_input(image, version=2)
    #     label_batch = oracle.predict(x_train)
    #     y_train = np.argmax(label_batch, axis=1)
    #     return x_train, y_train
    #
    # ds = ds_images.map(preprocess)

def validation_generator(oracle, validation_dir, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_it = datagen.flow_from_directory(validation_dir, class_mode=None,
                                         batch_size=batch_size,
                                         shuffle=True, target_size=(224, 224))
    while True:
        im_batch = val_it.next()
        x_val = utils.preprocess_input(im_batch, version=2)
        label_batch = oracle.predict(x_val)
        y_val = np.argmax(label_batch, axis=1)
        yield x_val, y_val

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
