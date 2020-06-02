import tensorflow as tf
import numpy as np
from attacks.blackbox.adversaries import generate_adversarial_sample, run_fgsm_attack
from attacks.blackbox.utilities import sess
from attacks.blackbox.models import load_model
from attacks.blackbox.params import VALIDATION_SET
from matplotlib import pyplot as plt


if __name__ == '__main__':

    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_it = datagen.flow_from_directory(VALIDATION_SET, class_mode='sparse', batch_size=4,
                                           shuffle=True, target_size=(224, 224))

    def gen():
        while True:
            x_train, y_train = train_it.next()
            yield x_train.astype(np.float32) / 255.0, y_train.astype(np.int)

    train_ds = gen()

    lambdas = [0.1, 0.15]
    num_iter = 500

    model = load_model('resnet50', num_classes=231, trained=True,
                       weights_path='/cs/ep/503/dataset/weights/substitute_resnet50_4.h5')

    with sess:
        for batch, _ in train_ds:
            adv_batch = generate_adversarial_sample(model, batch, run_fgsm_attack, {},
                                                    sess=tf.get_default_session())
            for i in range(len(batch)):
                fig, axarr = plt.subplots(1, 2)
                axarr[0].imshow(batch[i])
                axarr[1].imshow(adv_batch[i])
                plt.show()
            break
