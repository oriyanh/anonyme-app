import os
import tensorflow as tf
from attacks.blackbox.vgg_blackbox import Vggface2_ResNet50
from attacks.blackbox.substitute_model import SubstituteModel
import attacks.blackbox.params as params
from attacks.blackbox.augmentation import augment_dataset

optimizer = tf.keras.optimizers.Adam(params.LEARNING_RATE, beta_1=params.MOMENTUM)
loss_obj_oracle = tf.keras.losses.sparse_categorical_crossentropy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

GPU_CONST = '2'


def init_black_box_vggface2_keras(weights_path=params.VGGFACE2_BLACKBOX_WEIGHTS_PATH):
    """
    Loads trained VGGFace2 Keras black-box model using provided weights_path
    *Ripped straight out of vggface2_Keras predict module*
    :param weights_path: Path containing VGGFace2 Keras model weights
    :type weights_path: str
    :return: Loaded model
    """

    # Set basic environments.
    # Initialize GPUs
    # toolkits.initialize_GPU(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CONST
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)  # TODO: Check if this is necessary if the model isn't even training

    # ==> loading the pre-trained model.
    model_eval = Vggface2_ResNet50()

    if os.path.isfile(weights_path):
        model_eval.load_weights(weights_path, by_name=True)
        print('==> successfully loaded the model {}'.format(weights_path))
    else:
        raise IOError('==> can not find the model to load {}'.format(weights_path))
    return model_eval


def train(oracle, epochs_substitute, epochs_training, batch_size):
    x_train, y_train = params.load_initial_set(params.NUM_INIT_SAMPLES)

    for epoch in range(epochs_substitute):
        substitute = SubstituteModel()
        estimated_labels = oracle(x_train)
        train_substitute(substitute, epochs_training, x_train, estimated_labels, batch_size)
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

    return train_step


if __name__ == '__main__':
    init_black_box_vggface2_keras()
