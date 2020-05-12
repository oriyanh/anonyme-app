import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras_vggface import utils, VGGFace

import attacks.blackbox.params as params
# from attacks.blackbox.blackbox_model import graph, sess
from attacks.blackbox.models import graph, sess, save_model, load_model


loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

def substitute_model(num_classes):
    model = tf.keras.Sequential(layers=[Conv2D(64, 2), MaxPool2D(2), Conv2D(64, 2),
                                        MaxPool2D(2), Flatten(), Dense(200, activation='sigmoid'),
                                        Dense(200, activation='sigmoid'), Dense(100, activation='relu'),
                                        Dense(num_classes, activation='softmax')])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model_type, oracle, train_dir, validation_dir, num_epochs, batch_size):
    model = load_model(model_type=model_type, trained=False)
    train_ds, nsamples = get_training_set(oracle, train_dir, batch_size)
    nsteps = (nsamples // batch_size) + 1

    ### This block of code will be used instead of `fit()` when we get rid of Keras ###
    for epoch in range(num_epochs):
        print(f"Starting training epoch #{epoch + 1}")
        epoch_loss = 0.
        epoch_acc = 0.
        step = 0
        for im_batch, label_batch in train_ds:
            if step >= nsteps:
                break
            [loss, acc] = model.train_on_batch(im_batch, label_batch)
            epoch_loss += loss
            epoch_acc += acc
            print(f"Step {step+1}/{nsteps} ({100 * (step+1) / nsteps:.2f}%) - Loss={loss}, accuracy={acc:.3f}")
            step += 1
        epoch_loss /= nsteps
        epoch_acc /= nsteps
        print(f"Average training loss for epoch: {epoch_loss} ; Average accuracy: {epoch_acc:.3f}")
        print("Saving checkpoint")
        save_model(model, 'resnet50')

    ### End block ###

    val_ds, _ = get_training_set(oracle, validation_dir, batch_size)
    validation_acc = 0.
    num_validation_steps = 10
    for i in range(num_validation_steps):
        images, y_true = next(val_ds)
        pred = model.predict(images)
        y_pred = np.argmax(pred, axis=1)
        validation_acc += np.count_nonzero(y_pred == y_true)
    print(f"Substitute model accuracy after {num_epochs} epochs: {validation_acc/(num_validation_steps*batch_size):.2f}")
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
            yield x_train.astype(np.float32)/255.0, y_train

    ds_images = tf.data.Dataset.from_generator(gen, output_shapes=([None, 224, 224, 3], [None]),
                                               output_types=(tf.float32, tf.int32))
    # return ds_images, train_it.n
    return gen(), train_it.n

##### Useful implementations that we might want to use further down the line #####
##### Uncomment if needed

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
