import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras_vggface import utils, VGGFace

import attacks.blackbox.params as params
# from attacks.blackbox.blackbox_model import graph, sess
from attacks.blackbox.models import graph, sess
from attacks.blackbox.squeezenet import squeeze_net


loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

def substitute_model(num_classes):
    model = tf.keras.Sequential(layers=[Conv2D(64, 2), MaxPool2D(2), Conv2D(64, 2),
                                        MaxPool2D(2), Flatten(), Dense(200, activation='sigmoid'),
                                        Dense(200, activation='sigmoid'), Dense(100, activation='relu'),
                                        Dense(num_classes, activation='softmax')])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def vggface(num_classes):
    model = VGGFace(include_top=True, model='resnet50', weights=None, classes=num_classes)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model(weights_path, num_classes, model='squeezenet'):
    if 'squeezenet' == model:
        model = squeeze_net(num_classes)
    elif 'custom' == model:
        model = substitute_model(num_classes)
    else:
        raise NotImplementedError(f"Unspoorted model architecture {model}")

    model.build(input_shape=[None, 224, 224, 3])
    model.load_weights(weights_path)
    assert model.predict(np.random.randn(1, 224, 224, 3)) is not None
    print("Model loaded successfully!")
    return model


def save_model(model, weights_path):
    model.save_weights(weights_path, save_format='h5')


def train(model, oracle, train_dir, validation_dir, num_epochs, batch_size):
    train_ds, nsamples = get_training_set(oracle, train_dir, batch_size)
    nsteps = (nsamples // batch_size) + 1
    model.fit(train_ds, epochs=num_epochs, steps_per_epoch=nsteps, verbose=1)

    ### This block of code will be used instead of `fit()` when we get rid of Keras ###
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
    ### End block ###

    val_ds, _ = get_training_set(oracle, validation_dir, batch_size)
    [loss, accuracy] = model.evaluate(val_ds, steps=10)
    print(f"Validation loss: {loss:.2f} ; Validation Accuracy: {accuracy:.2f}")
    return model


def get_training_set(oracle, train_dir, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_it = datagen.flow_from_directory(train_dir, class_mode=None, batch_size=batch_size,
                                           shuffle=True, target_size=(224, 224))

    def gen():
        while True:
            x_train = train_it.next()
            im_batch_norm = utils.preprocess_input(x_train, version=2)
            with graph.as_default():
                tf.keras.backend.set_session(sess)
                label_batch = oracle.predict(im_batch_norm)
            y_train = np.argmax(label_batch, axis=1)
            yield x_train/255.0, y_train

    ds_images = tf.data.Dataset.from_generator(gen, output_shapes=([None, 224, 224, 3], [None]),
                                               output_types=(tf.float32, tf.int32))
    return ds_images, train_it.n

##### Useful implementations that we might want to use further down the line #####
##### Uncomment if needed

# def training_generator(oracle, train_dir, batch_size):
#     datagen = tf.keras.preprocessing.image.ImageDataGenerator()
#     train_it = datagen.flow_from_directory(train_dir, class_mode=None,
#                                            batch_size=batch_size,
#                                            shuffle=True, target_size=(224, 224))
#     nsteps = (len(train_it.filepaths) // batch_size) + 1
#
#     def gen():
#         while True:
#             im_batch = train_it.next()
#             x_train = utils.preprocess_input(im_batch, version=2)
#             with graph.as_default():
#                 tf.keras.backend.set_session(sess)
#                 label_batch = oracle.predict(x_train)
#             y_train = np.argmax(label_batch, axis=1)
#             yield x_train, y_train
#
#     return gen, nsteps
#
# def validation_generator(oracle, validation_dir, batch_size):
#     datagen = tf.keras.preprocessing.image.ImageDataGenerator()
#     val_it = datagen.flow_from_directory(validation_dir, class_mode=None,
#                                          batch_size=batch_size,
#                                          shuffle=True, target_size=(224, 224))
#     while True:
#         im_batch = val_it.next()
#         x_val = utils.preprocess_input(im_batch, version=2)
#         label_batch = oracle.predict(x_val)
#         y_val = np.argmax(label_batch, axis=1)
#         yield x_val, y_val
#
# def get_train_step():
#     @tf.function
#     def train_step(model, images, labels):
#         with tf.GradientTape() as tape:
#             preds = model(images)
#             loss = loss_obj(labels, preds)
#
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         train_loss(loss)
#         train_accuracy(labels, preds)
#
#     return train_step
