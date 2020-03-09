"""
This model uses MTCNN to align all test images
"""

import subprocess
import os
from project_params import ROOT_DIR

MTCNN_PATH = "mtcnn"
MTCNN_ALIGN_FNAME = "align_dataset_mtcnn.py"


def align(venv_dir, input_dir, output_dir, image_size):
    """

    :param venv_dir:
    :param input_dir:
    :param output_dir:
    :param image_size:
    :return:
    """

    if not (os.path.exists(venv_dir) and os.path.exists(input_dir)):
        print("Invalid path received")
        return

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    subprocess.run(f"./align.sh {venv_dir} {os.path.join(ROOT_DIR, MTCNN_PATH, MTCNN_ALIGN_FNAME)} "
                   f"{input_dir} {output_dir} {image_size}", shell=True)


if __name__ == '__main__':
    align("/cs/ep/503/temp_venv", "/cs/ep/503/vggface2/vggface2_test/test",
          "/cs/ep/503/vggface2/vggface2_test/test_aligned", 160)
