import click

from project_params import ROOT_DIR

import os

LPIPS_THRESHOLD_VAL = 0.2

os.umask(2)

import sys

sys.path.append(os.path.join(ROOT_DIR, 'PerceptualSimilarity'))

from datetime import datetime
from PIL import Image
from matplotlib import pyplot as plt
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


def analyze_statistics(df, step_size, image_num, output_dir, model_name):
    init_images = df[df['iter_num'] == 0]
    bb_misclassified = df[df['bb_is_same'] == False]

    x_axis = np.arange(0, 0.201, 0.005)

    # Cumulative percentage of blackbox misclassifications by LPIPS dist
    misclassified_counts = [float(len(bb_misclassified[
                                          bb_misclassified['lpips_dist'] <= i])) / image_num
                            for i in x_axis]
    plt.figure()
    plt.plot(x_axis, misclassified_counts)
    plt.title(f"{model_name}\nCumulative misclassification rate against LPIPS distance")
    plt.xlabel("LPIPS dist. from initial image")
    plt.ylabel("Misclassification rate")
    plt.savefig(os.path.join(output_dir,
                             f'{model_name} - eps_{step_size:.3f}__percentage_cumulative.jpg'))

    # Blackbox misclassifications success rate as a function of initial blackbox confidence
    conf_bins = np.arange(0, 1.01, 0.1)

    init_images_conf_count = []
    bb_misclassified_conf_count = []
    for conf_bin in conf_bins:
        init_images_conf_count.append(len(init_images[init_images['bb_init_conf'].round(1) == conf_bin]))
        bb_misclassified_conf_count.append(len(bb_misclassified[bb_misclassified['bb_init_conf'].round(1) == conf_bin]))

    init_images_conf_count = np.array(init_images_conf_count, dtype=np.float)
    bb_misclassified_conf_count = np.array(bb_misclassified_conf_count, dtype=np.float)

    conf_count_data = [bb_misclassified_conf_count, init_images_conf_count]
    conf_count_labels = ['Misclassification count', 'Image count']

    conf_success_rates = np.zeros_like(bb_misclassified_conf_count, dtype=np.int)
    conf_success_rates[init_images_conf_count != 0] = (100 * bb_misclassified_conf_count[init_images_conf_count != 0] /
                                                       init_images_conf_count[init_images_conf_count != 0]).astype(np.int)

    color_list = ['r', 'g']
    gap = .08 / len(conf_count_data)

    plt.figure()
    plt.title(f'{model_name}\nMisclassification rate by initial blackbox confidence')
    for i, row in enumerate(conf_count_data):
        plt.bar(conf_bins + i * gap, row, width=gap, color=color_list[i % len(color_list)],
                label=conf_count_labels[i])
    plt.xticks(conf_bins)
    plt.legend()
    plt.xlabel("Initial image confidence %")
    plt.ylabel("# of images")
    for i, conf_success_rate in enumerate(conf_success_rates):
        if not conf_success_rate:
            continue
        plt.text(conf_bins[i], init_images_conf_count[i] + .5, s=f'{conf_success_rate}%')
    plt.savefig(os.path.join(output_dir,
                             f'{model_name} - eps_{step_size:.3f}_misscl_by_conf_success_rate_bar.jpg'))

    # Average distances as a function of iteration numbers
    mean_df = df[['iter_num', 'lpips_dist', 'l1_dist', 'l2_dist']].groupby(['iter_num']).mean()  # OMG SO MEAN
    metric_labels = ['LPIPS', 'L1', 'L2']
    for i in range(3):
        plt.figure()
        plt.xticks(np.arange(len(mean_df), step=5))
        plt.xlabel("# of Attack Iteration")
        plt.ylabel("Avg. distance")
        plt.title(f"{model_name}\nAverage {metric_labels[i]} dist. by # of attack iteration")
        plt.plot(mean_df.iloc[:, i])
        plt.savefig(os.path.join(output_dir,
                                 f'{model_name} - eps_{step_size:.3f}_average_{mean_df.columns[i]}_by_iter.jpg'))

    # Success rate by class ID
    lpips_success_thresholds = [.1, .2]
    files_per_class = init_images.groupby(['bb_init_pred'])['file_name'].count()
    for lpips_success_threshold in lpips_success_thresholds:
        misclassifications_per_class = bb_misclassified[bb_misclassified[
            'lpips_dist'] < lpips_success_threshold].groupby(['bb_init_pred'])['file_name'].count()
        success_rate = misclassifications_per_class.divide(files_per_class, fill_value=0)

        plt.figure()
        plt.title(f'{model_name}\nSuccess rate by class ID for LPIPS threshold {lpips_success_threshold}')
        # plt.bar(os.listdir(dataset_dir), success_rate)
        success_rate.plot.bar()
        plt.yticks(np.arange(0., 1.01, .1))
        plt.xlabel("Class ID")
        plt.ylabel("Success rate")
        plt.savefig(os.path.join(output_dir,
                                 f'{model_name} - eps_{step_size:.3f}_misscl_success_rate_by_class.jpg'))

@click.command()
@click.argument('architecture', type=click.Choice(['resnet50', 'squeezenet', 'senet50']))
@click.argument('label')
@click.argument('eval_dataset', type=click.Path(exists=True))
@click.option('--batch-size', default=4, help='Evaluation batch size')
@click.option('--step-size', default=0.004, help='FGSM attack step size')
@click.option('--max-iter', default=50, help='Max FGSM iterations')
@click.option('--normalize-images', is_flag=True)
@click.option('--cross', is_flag=True)
def evaluate_attack(architecture, label, eval_dataset, batch_size, step_size, max_iter, normalize_images, cross):

    global DESTINATION_PATH
    DESTINATION_PATH = f'{DESTINATION_PATH}-{label}'

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

    blackbox = load_model('blackbox', architecture=architecture)
    if cross:
        sub_architecture = 'resnet50' if architecture == 'senet50' else 'senet50'
        substitute = load_model('blackbox', architecture=sub_architecture)
    else:
        substitute = blackbox

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

        analyze_statistics(df, step_size, val_it.n, DESTINATION_PATH,
                           label)


if __name__ == '__main__':
    evaluate_attack()
