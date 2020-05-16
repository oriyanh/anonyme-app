import tensorflow as tf
import attacks.blackbox.params as params
from attacks.blackbox import models
from attacks.blackbox.augmentation import augment_dataset
from attacks.blackbox.utilities import get_training_set, predict_and_save, sess


def train(oracle, substitute_type, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0 and nepochs_training > 0
    # image_dir = params.TRAIN_SET_ALIGNED
    image_dir = params.TRAIN_SET_INITIAL
    # validation_dir = train_dir
    model = None
    for epoch_sub in range(1, nepochs_substitute + 1):
        print(f"Starting training on new substitute model, epoch #{epoch_sub}")

        print("a) Acquiring oracle predictions")
        predict_and_save(oracle, image_dir, params.TRAIN_SET_WORKING, batch_size)
        train_dir = params.TRAIN_SET_WORKING

        print("b) Start training substitute model")
        train_ds, nsteps, num_classes = get_training_set(train_dir, batch_size)
        model = models.load_model(model_type=substitute_type, trained=False, num_classes=num_classes)

        for epoch in range(nepochs_training):

            print(f"b.1) Starting training epoch #{epoch + 1}")
            epoch_loss = 0.
            epoch_acc = 0.
            step = 0
            for im_batch, label_batch in train_ds:
                if step >= nsteps:
                    break
                [loss, acc] = model.train_on_batch(im_batch, label_batch)
                epoch_loss += loss
                epoch_acc += acc
                print(
                    f"Step {step + 1}/{nsteps} ({100 * (step + 1) / nsteps:.2f}%) - Loss={loss}, accuracy={acc}")
                step += 1
            epoch_loss /= nsteps
            epoch_acc /= nsteps
            print(f"Average training loss for epoch: {epoch_loss} ; Average accuracy: {epoch_acc}")
            print("b.2) Saving checkpoint")
            models.save_model(model, substitute_type)
        if epoch_sub < nepochs_substitute:
            print("c) Augmenting")
            model = models.load_model(model_type=substitute_type, trained=False, num_classes=num_classes)
            image_dir = augment_dataset(model, train_dir, params.LAMBDA)
    return model

PRE_TRAINED = False

def main():
    oracle = models.load_model(model_type='blackbox', architecture='resnet50')
    if PRE_TRAINED:
        substitute_model = models.load_model(model_type='resnet50', trained=True)
    else:
        substitute_model = train(oracle, 'resnet50', params.EPOCHS_SUBSTITUTE,
                                 params.EPOCHS_TRAINING, params.BATCH_SIZE)

if __name__ == '__main__':
    tf.keras.backend.set_session(sess)
    main()
    sess.close()
