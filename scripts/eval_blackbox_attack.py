import click

from project_params import ROOT_DIR

import os

from scripts.eval_stat_analyzer import analyze_statistics

LPIPS_THRESHOLD_VAL = 0.2

os.umask(2)

import sys

sys.path.append(os.path.join(ROOT_DIR, 'PerceptualSimilarity'))

from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from skimage.measure import compare_ssim
import cv2

from PerceptualSimilarity.models import PerceptualLoss
from project_utilities import perceptual_loss
from attacks.blackbox.adversaries import fgsm, calc_preds
from attacks.blackbox.utilities import sess, standardize_batch
from attacks.blackbox.models import load_model
from attacks.blackbox.params import DATASET_BASE_PATH

TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S%f')
DESTINATION_PATH = os.path.join(DATASET_BASE_PATH, 'attack_eval_res', TIMESTAMP)


def get_img_dest_path(blackbox_pred, filename):
    return os.path.join(DESTINATION_PATH, f"n{str(blackbox_pred).rjust(6, '0')}", filename.split('_', 1)[0])


def save_image(im, blackbox_pred, filename):
    img_dest_path = get_img_dest_path(blackbox_pred, filename)
    if not os.path.exists(img_dest_path):
        os.makedirs(img_dest_path)
    Image.fromarray(im).save(os.path.join(img_dest_path, filename))


def calc_im_diff(im, res_im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    res_im_gray = cv2.cvtColor(res_im, cv2.COLOR_BGR2GRAY)
    _, diff = compare_ssim(im_gray, res_im_gray, full=True)
    diff = (diff * 255).astype(np.uint8)
    return diff


def evaluate_fgsm(substitute, blackbox, im_batch, labels, batch_filenames, batch_ph,
                  label_ph, adv_im_batch, lpips, sess=None, eps=0.05, num_iter=100, attack_bounds=(0., 1.),
                  preprocess_func=None):
    if sess is None:
        sess = tf.get_default_session()

    batch_rows = []
    batch_index = []

    res_im_batch = np.copy(im_batch)
    adv_diff_batch = np.zeros_like(res_im_batch)

    bb_orig_preds, bb_orig_conf = calc_preds(blackbox, res_im_batch)
    bb_preds, bb_conf = bb_orig_preds, bb_orig_conf

    prev_bb_preds = bb_preds

    sub_orig_preds, sub_orig_conf = calc_preds(substitute, res_im_batch)
    sub_preds, sub_conf = sub_orig_preds, sub_orig_conf
    attack_bounds_diff = attack_bounds[1] - attack_bounds[0]
    threshold_mask = np.ones(im_batch.shape[0], np.bool)

    for i in range(num_iter + 1):

        lpips_dists = perceptual_loss(lpips, im_batch, res_im_batch)
        l1_dists = np.sum(np.abs(res_im_batch - im_batch), axis=(1, 2, 3))
        l2_dists = np.sqrt(np.sum((res_im_batch - im_batch) ** 2, axis=(1, 2, 3)))

        for j in range(len(res_im_batch)):

            filename = f"{batch_filenames[j].split('.', 1)[0]}_iter_{i}_eps_{eps:.3f}.jpg"

            if threshold_mask[j]:
                save_image((res_im_batch[j] * (255. / attack_bounds_diff)).astype(np.uint8),
                           labels[j],
                           filename)

                batch_rows.append([
                    i,
                    bb_conf[j],
                    bb_preds[j],
                    bb_preds[j] == bb_orig_preds[j],
                    sub_conf[j],
                    sub_preds[j],
                    sub_preds[j] == sub_orig_preds[j],
                    bb_orig_preds[j],
                    bb_orig_conf[j],
                    lpips_dists[j],
                    l1_dists[j],
                    l2_dists[j],
                    os.path.join(get_img_dest_path(labels[j], filename), filename),
                    ])
                batch_index.append(os.path.join(f"n{str(labels[j]).rjust(6, '0')}", batch_filenames[j]))

            # if prediction has changed, save diff
            if prev_bb_preds[j] != bb_preds[j]:
                diff_filename = f"{filename}_diff.jpg"
                inv_diff_filename = f"{filename}_diff_inv.jpg"
                diff = calc_im_diff(im_batch[j], res_im_batch[j])
                save_image(diff, labels[j], diff_filename)
                save_image(255 - diff, labels[j], inv_diff_filename)

        threshold_mask = np.logical_and(lpips_dists < LPIPS_THRESHOLD_VAL, bb_preds == bb_orig_preds)

        if i == num_iter or not np.any(threshold_mask):
            break

        print(f"Iteration #{i + 1}/{num_iter}")

        adv_diff_batch[threshold_mask] = sess.run(
            adv_im_batch,
            feed_dict={
                batch_ph: (preprocess_func(res_im_batch) if preprocess_func else res_im_batch)[threshold_mask],
                label_ph: sub_orig_preds[threshold_mask],
            })

        res_im_batch[threshold_mask] += adv_diff_batch[threshold_mask]
        res_im_batch = np.clip(res_im_batch, attack_bounds[0], attack_bounds[1])

        prev_bb_preds = bb_preds

        bb_preds, bb_conf = calc_preds(blackbox, res_im_batch)
        sub_preds, sub_conf = calc_preds(substitute, res_im_batch)
    return batch_rows, batch_index


@click.command()
@click.argument('sub_architecture', type=click.Choice(['resnet50', 'squeezenet', 'senet50']))
@click.argument('sub_weights', type=click.Path(exists=True))
@click.argument('sub_classes', type=click.INT)
@click.argument('sub_label')
@click.argument('eval_dataset', type=click.Path(exists=True))
@click.option('--blackbox-architecture', default=None, type=click.Choice(['resnet50', 'senet50']))
@click.option('--batch-size', default=4, help='Evaluation batch size')
@click.option('--step-size', default=0.004, help='FGSM attack step size')
@click.option('--max-iter', default=50, help='Max FGSM iterations')
@click.option('--normalize-images', is_flag=True)
def evaluate_attack(sub_architecture, sub_weights, sub_classes, sub_label, eval_dataset,
                    blackbox_architecture, batch_size, step_size, max_iter, normalize_images):

    global DESTINATION_PATH
    DESTINATION_PATH = f'{DESTINATION_PATH}-{sub_label}'

    if blackbox_architecture is None:
        if sub_architecture == 'squeezenet':
            raise ValueError('Blackbox architecture must be specified for squeezenet substitute')
        blackbox_architecture = sub_architecture

    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_it = datagen.flow_from_directory(eval_dataset,
                                         shuffle=False, class_mode='sparse', batch_size=batch_size,
                                         target_size=(224, 224))

    nbatches = val_it.n // batch_size
    if nbatches * batch_size < val_it.n:
        nbatches += 1

    class_map = {idx: int(name[1:]) for name, idx in val_it.class_indices.items()}
    vectorized_get = np.vectorize(class_map.get)

    lpips = PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())

    def gen():
        while True:
            x_val, y_unmapped = val_it.next()
            y_mapped = vectorized_get(y_unmapped)
            idx = (val_it.batch_index - 1) * (val_it.batch_size)
            batch_filenames = [os.path.basename(filename)
                               for filename in val_it.filenames[idx:
                                                                (idx + val_it.batch_size if idx > 0 else None)]]
            yield x_val.astype(np.float32) / (255.0 if normalize_images else 1.0), \
                  y_mapped.astype(np.int), batch_filenames

    train_ds = gen()
    attack_bounds = (0., 255. / (255. if normalize_images else 1.))

    substitute = load_model(sub_architecture, num_classes=sub_classes, trained=True,
                            weights_path=sub_weights)
    blackbox = load_model('blackbox', architecture=blackbox_architecture)

    with sess:
        batch_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                                  name="batch_in")
        label_ph = tf.placeholder(tf.int32, shape=[None], name="pred_in")

        print(f"Evaluating using step size {step_size:.3f}")
        stat_rows = []
        stat_index = []
        adv_im_batch = fgsm(substitute, batch_ph, label_ph, step_size * (attack_bounds[1] - attack_bounds[0]))

        for step_num, (batch, labels, filenames) in enumerate(train_ds):
            if step_num >= nbatches:
                break
            print(f"Evaluating batch #{step_num + 1}/{nbatches}")

            batch_rows, batch_index = evaluate_fgsm(substitute, blackbox, batch, labels, filenames, batch_ph,
                                                    label_ph, adv_im_batch, lpips, preprocess_func=standardize_batch,
                                                    eps=step_size, num_iter=max_iter, attack_bounds=attack_bounds)
            stat_rows += batch_rows
            stat_index += batch_index

        df = pd.DataFrame(stat_rows,
                          columns=[
                              'iter_num',
                              'bb_conf',
                              'bb_pred',
                              'bb_is_same',
                              'sub_conf',
                              'sub_pred',
                              'sub_is_same',
                              'bb_init_pred',
                              'bb_init_conf',
                              'lpips_dist',
                              'l1_dist',
                              'l2_dist',
                              'file_name',
                          ],
                          index=stat_index)
        res_file_name = f'eps_{step_size:.3f}_res.csv'
        df.to_csv(os.path.join(DESTINATION_PATH, res_file_name))

        print(f"Created result file {res_file_name}")

        analyze_statistics(df, DESTINATION_PATH, sub_label)


if __name__ == '__main__':
    evaluate_attack()
