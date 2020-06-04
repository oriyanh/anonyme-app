import os
import csv
import tensorflow as tf
import numpy as np
from attacks.blackbox.utilities import

from attacks.blackbox.adversaries import fgsm, calc_preds
from attacks.blackbox.utilities import sess
from attacks.blackbox.models import load_model
from attacks.blackbox.params import VALIDATION_SET


def evaluate_fgsm(model, im_batch, blackbox_preds, batch_filenames, sess=tf.get_default_session(),
                    eps=0.05, num_iter=100):

    # save_img_fn = lambda x, y: Image.fromarray(x).save(
    #     os.path.join(new_image_dir, f"{y}.jpg".rjust(11, "0")))

    res_im_batch = np.copy(im_batch)
    batch_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                              name="batch_in")
    label_ph = tf.placeholder(tf.int32, shape=[None], name="pred_in")
    adv_im_batch = fgsm(model, batch_ph, label_ph, eps)

    orig_preds = calc_preds(model, res_im_batch)
    preds = orig_preds

    for i in range(num_iter):
        # if i + 1 % 10 == 0:

        print(f"Iteration {i + 1}/{num_iter}")
        res_im_batch = sess.run(
            adv_im_batch,
            feed_dict={
                batch_ph: res_im_batch,
                label_ph: preds,
            })
        preds = calc_preds(model, res_im_batch)

    return res_im_batch



if __name__ == '__main__':

    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_it = datagen.flow_from_directory(VALIDATION_SET, class_mode='sparse', batch_size=4,
                                         target_size=(224, 224))

    class_map = {idx: int(name[1: ]) for name, idx in val_it.class_indices.items()}
    vectorized_get = np.vectorize(class_map.get)

    def gen():
        while True:
            x_val, y_unmapped = val_it.next()
            y_mapped = vectorized_get(y_unmapped)
            idx = (val_it.batch_index - 1) * (val_it.batch_size)
            batch_filenames = [os.path.basename(filename)
                               for filename in val_it.filenames[idx: idx + val_it.batch_size]]
            yield x_val.astype(np.float32) / 255.0, y_mapped.astype(np.int), batch_filenames


    train_ds = gen()

    step_sizes = [0.025, 0.05]
    num_iter = 500

    model = load_model('resnet50', num_classes=231, trained=True,
                       weights_path='/cs/ep/503/dataset/weights/substitute_resnet50_4.h5')

    with sess:
        for step_size in step_sizes:
            for batch, labels, filenames in train_ds:
                # adv_batch = generate_adversarial_sample(model, batch, run_fgsm_attack, {},
                #                                         sess=tf.get_default_session())
                for i in range(len(batch)):
                    fig, axarr = plt.subplots(1, 2)
                    axarr[0].imshow(batch[i])
                    axarr[1].imshow(batch[i])
                    # axarr[1].imshow(adv_batch[i])
                    plt.show()
                break
