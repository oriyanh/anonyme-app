import os
import click
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE, labelsize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


def analyze_statistics(df, output_dir, model_name, debug=False):
    init_images = df[df['iter_num'] == 0]
    bb_misclassified = df[df['bb_is_same'] == False]

    x_axis = np.arange(0, 0.201, 0.005)

    # Cumulative percentage of blackbox misclassifications by LPIPS dist
    misclassified_counts = [float(len(bb_misclassified[
                                          bb_misclassified['lpips_dist'] <= i])) / len(init_images)
                            for i in x_axis]
    plt.figure()
    plt.plot(x_axis, misclassified_counts, linewidth=0.75, color='black')
    # plt.title(f"{model_name}\nCumulative misclassification rate by perturbation budget")
    plt.xlabel("Perceptual budget")
    plt.ylabel("Misclassification rate")
    if debug:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir,
                                 f'{model_name} - percentage_cumulative.jpg'))

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
    conf_count_labels = ['Misclassified images', 'All images']

    conf_success_rates = np.zeros_like(bb_misclassified_conf_count, dtype=np.int)
    conf_success_rates[init_images_conf_count != 0] = (100 * bb_misclassified_conf_count[init_images_conf_count != 0] /
                                                       init_images_conf_count[init_images_conf_count != 0]).astype(np.int)

    color_list = ['r', 'g']
    gap = .08 / len(conf_count_data)

    plt.figure()
    # plt.title(f'{model_name}\nMisclassification rate by initial blackbox confidence')
    for i, row in enumerate(conf_count_data):
        plt.bar(conf_bins + i * gap, row, width=gap, color=color_list[i % len(color_list)],
                label=conf_count_labels[i])
    plt.xticks(conf_bins)
    plt.legend()
    plt.xlabel("Blackbox prediction confidence")
    plt.ylabel("Image count")
    for i, conf_success_rate in enumerate(conf_success_rates):
        if not conf_success_rate:
            continue
        plt.text(conf_bins[i], init_images_conf_count[i] + 2.5, s=f'{conf_success_rate}%')
    if debug:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir,
                                 f'{model_name} - misscl_by_conf_success_rate_bar.jpg'))

    # Average distances as a function of iteration numbers
    mean_df = df[['iter_num', 'lpips_dist', 'l1_dist', 'l2_dist']].groupby(['iter_num']).mean()  # OMG SO MEAN
    metric_labels = ['LPIPS', 'L1', 'L2']
    for i in range(3):
        plt.figure()
        plt.xticks(np.arange(len(mean_df), step=5))
        plt.xlabel("Attack Iteration")
        plt.ylabel("Avg. distance")
        # plt.title(f"{model_name}\nAverage {metric_labels[i]} dist. by # of attack iteration")
        plt.plot(mean_df.iloc[:, i], linewidth=0.75, color='black')
        if debug:
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir,
                                     f'{model_name} - average_{mean_df.columns[i]}_by_iter.jpg'))

    # Success rate by class ID
    lpips_success_thresholds = [.1, .2]
    files_per_class = init_images.groupby(['bb_init_pred'])['file_name'].count()
    for lpips_success_threshold in lpips_success_thresholds:
        misclassifications_per_class = bb_misclassified[bb_misclassified[
                                                            'lpips_dist'] < lpips_success_threshold].groupby(['bb_init_pred'])['file_name'].count()
        success_rate = misclassifications_per_class.divide(files_per_class, fill_value=0)

        plt.figure()
        # plt.title(f'{model_name}\nSuccess rate by class ID for LPIPS threshold {lpips_success_threshold}')
        success_rate.plot.bar()
        plt.yticks(np.arange(0., 1.01, .1))
        plt.xlabel("Class ID")
        plt.ylabel("Success rate")
        if debug:
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir,
                                     f'{model_name} - misscl_success_rate_by_class.jpg'))


@click.command()
@click.argument('csv_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_name')
@click.option('--debug', is_flag=True)
def analyze_from_csv(csv_path, model_name, debug):
    dest_path = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)
    analyze_statistics(df, dest_path, model_name, debug=debug)


if __name__ == '__main__':
    analyze_from_csv()
