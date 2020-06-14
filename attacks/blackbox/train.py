print(f"Loading module {__file__}")
import os
import numpy as np
os.umask(2)
import tensorflow as tf
import attacks.blackbox.params as params
from time import perf_counter
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
from attacks.blackbox import models
from attacks.blackbox.augmentation import augment_dataset
from attacks.blackbox.utilities import get_train_set, oracle_classify_and_save, sess, get_validation_set


def train(oracle, substitute_type, nepochs_substitute, nepochs_training, batch_size):
    assert nepochs_substitute > 0 and nepochs_training > 0

    train_set_dir = params.TRAIN_SET
    validation_dir = params.VALIDATION_SET
    train_dir = params.TRAIN_SET_WORKING

    print("1) Preprocess dataset - acquire oracle predictions and prune")
    # oracle_classify_and_save(oracle, train_set_dir, train_dir, batch_size, prune_threshold=10)

    model = None
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch_sub in range(1, nepochs_substitute + 1):
        print(f"2) Training substitute model #{epoch_sub}/{nepochs_substitute}")
        train_ds, nsteps_train, num_classes, class_indices = get_train_set(train_dir, batch_size)
        model = models.load_model(model_type=substitute_type, trained=False, num_classes=num_classes)
        for epoch in range(nepochs_training):
            print(f"2.1) Training epoch #{epoch + 1}")
            epoch_start_time = perf_counter()
            epoch_loss = 0.
            epoch_acc = 0.
            step_train = 0
            for im_batch, label_batch in train_ds:
                if step_train >= nsteps_train:
                    break
                [loss, acc] = model.train_on_batch(im_batch, label_batch)
                epoch_loss += loss
                epoch_acc += acc
                step_train += 1
                print(f"Step {step_train}/{nsteps_train} ({100 * (step_train) / nsteps_train:.2f}%) "
                      f"- Loss={loss}, accuracy={acc}; ", end="")

                time_now = perf_counter()
                time_elapsed = time_now - epoch_start_time
                time_per_step = time_elapsed / step_train
                steps_remaining = nsteps_train - step_train
                time_remaining = steps_remaining * time_per_step
                print(f"Est. time remaining for epoch: {timedelta(seconds=time_remaining)}")

            epoch_loss /= nsteps_train
            epoch_acc /= nsteps_train
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            print(f"Average training loss: {epoch_loss} ; Average accuracy: {epoch_acc}")

            # TODO Validation dir is assumed to be mapped using oracle predictions
            print(f"2.2) Validation epoch #{epoch + 1}")
            validation_ds, nsteps_val = get_validation_set(validation_dir, class_indices, batch_size)
            step_val = 0
            validation_loss = 0.
            validation_acc = 0.
            for im_batch, label_batch in validation_ds:
                if step_val >= nsteps_val:
                    break
                [loss, acc] = model.test_on_batch(im_batch, label_batch)
                validation_loss += loss
                validation_acc += acc
                step_val += 1
                print(f"Step {step_val}/{nsteps_val} ({100 * step_val / nsteps_val:.2f}%) "
                      f"- Loss={loss}, accuracy={acc}")

            validation_loss /= nsteps_val
            validation_acc /= nsteps_val
            val_losses.append(validation_loss)
            val_accuracies.append(validation_acc)
            print(f"Validation loss for epoch: {validation_loss} ; Validation accuracy: {validation_acc}")

            print("2.2) Save checkpoint")
            models.save_model(model, substitute_type, override=True)

        if epoch_sub < nepochs_substitute:
            print("3) Augment dataset")
            augmented_images_dir = augment_dataset(model, train_dir, params.LAMBDA)

            print("4) Acquire oracle predictions for new samples")
            oracle_classify_and_save(oracle, augmented_images_dir, train_dir, batch_size)

        print(f"Number of output classes in model #{nepochs_substitute}: {num_classes}")
        models.save_model(model, substitute_type, override=False)
        print("="*50)
    print("\n\nFinished training, generating graphs")
    graphs_path = os.path.join(params.PROJECT_DIR, "outputs", "graphs_resnet50", "accuracies", datetime.now().strftime("%Y%m%d%H%M%S%f"))
    os.makedirs(graphs_path)
    best_train_accuracy_idx = int(np.argmax(train_accuracies))
    best_val_accuracy_idx = int(np.argmax(val_accuracies))
    print(f"Epoch with best training accuracy: epoch #{best_train_accuracy_idx + 1}, value = {train_accuracies[best_train_accuracy_idx]:.3f}")
    print(f"Epoch with best validation accuracy: epoch #{best_val_accuracy_idx + 1}, value = {val_accuracies[best_val_accuracy_idx]:.3f}")
    plt.figure()
    plt.title(f"Training loss over {nepochs_training} epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Crossentropy Loss")
    plt.scatter(np.arange(1, nepochs_training+1).astype(np.int), train_losses)
    plt.savefig(os.path.join(graphs_path, "train_loss.jpg"))
    plt.figure()
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training accuracy over {nepochs_training} epochs")
    plt.scatter(np.arange(1, nepochs_training+1).astype(np.int), train_accuracies)
    plt.savefig(os.path.join(graphs_path, "train_accuracy.jpg"))
    plt.figure()
    plt.xlabel("# Epochs")
    plt.ylabel("Crossentropy Loss")
    plt.title(f"Validation loss over {nepochs_training} epochs")
    plt.scatter(np.arange(1, nepochs_training+1).astype(np.int), val_losses)
    plt.savefig(os.path.join(graphs_path, "val_loss.jpg"))
    plt.figure()
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Validation accuracy over {nepochs_training} epochs")
    plt.scatter(np.arange(1, nepochs_training+1).astype(np.int), val_accuracies)
    plt.savefig(os.path.join(graphs_path, "val_accuracy.jpg"))
    return model


print(f"Successfully loaded module {__file__}")

if __name__ == '__main__':
    tf.keras.backend.set_session(sess)
    oracle = None

    substitute_model = train(oracle, 'resnet50', params.EPOCHS_SUBSTITUTE,
                                 params.EPOCHS_TRAINING, params.BATCH_SIZE)
    sess.close()
