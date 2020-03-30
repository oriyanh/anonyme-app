import tensorflow as tf
from keras_vggface import utils
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

import attacks.blackbox.params as params
import numpy as np
from attacks.blackbox.blackbox_model import graph, sess


loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

def SubstituteModel(num_classes):
    model = tf.keras.Sequential(layers=[Conv2D(64, 2), MaxPool2D(2), Conv2D(64, 2),
                                        MaxPool2D(2), Flatten(), Dense(200, activation='sigmoid'),
                                        Dense(200, activation='sigmoid'), Dense(100, activation='relu'),
                                        Dense(num_classes, activation='softmax')])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model(weights_path, num_classes):
    model = SubstituteModel(num_classes)
    model.build(input_shape=[None, 224, 224, 3])
    model.load_weights(weights_path)
    return model

def save_model(model, weights_path):
    model.save_weights(weights_path, save_format='h5')

def train(model, oracle, train_dir, validation_dir, num_epochs, batch_size):
    # train_gen, nsamples = training_generator(oracle, train_dir, batch_size)
    train_ds, nsamples = get_training_set(oracle, train_dir, batch_size)
    train_step = get_train_step()
    nsteps = (nsamples // batch_size) + 1
    # model.fit_generator(iter(train_ds), epochs=num_epochs, steps_per_epoch=2, verbose=1)
    model.fit(train_ds, epochs=num_epochs, steps_per_epoch=nsteps, verbose=1)
    # for epoch in range(num_epochs):
    #     print(f"Start training epoch #{epoch + 1}")
    #     step = 0
    #     # for step in range(nsteps):
    #     for im_batch, label_batch in train_ds:
    #         if step >= nsteps:
    #                 break
    # #         model.train_on_batch(train_ds)
    #         train_step(model, im_batch, label_batch)
    #         print(f"Training epoch progress: step {step+1}/{nsteps} ({100 * (step+1) / nsteps:.2f}%)")
    #         step += 1
    #     print("Training loss: %s" % train_loss.result())
    #     print("Training accuracy: %s" % train_accuracy.result())
    #     train_loss.reset_states()
    #     train_accuracy.reset_states()

    val_ds, _ = get_training_set(oracle, validation_dir, batch_size)
    [loss, accuracy] = model.evaluate(val_ds, steps=10)
    print(f"Validation loss: {loss:.2f} ; Validation Accuracy: {accuracy:.2f}")
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

    ds_images = tf.data.Dataset.from_generator(gen, output_shapes=([None, 224, 224, 3], [None]),
                                               output_types=(tf.float32, tf.int32))
    return ds_images, train_it.n
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
        train_accuracy(labels, preds)

    return train_step
