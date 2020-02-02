import subprocess

PATH_MTCNN_SCRIPT = "/cs/ep/503/oriyan/repo/mtcnn/align_dataset_mtcnn.py"

def align(input_dir, output_dir, image_size):
    subprocess.run(f"python {PATH_MTCNN_SCRIPT} {input_dir} {output_dir} "
                          f"--image_size {image_size}", shell=True)