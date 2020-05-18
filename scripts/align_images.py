import sys
import os
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from PIL import Image

detector = MTCNN()

def extract_face(pixels, required_size=224):
    # detect faces in the image
    pixels_h, pixels_w, _ = pixels.shape
    results = detector.detect_faces(pixels)
    if not results:
        return None

    # extract the bounding box from the first face
    y1, x1, height, width = results[0]['box']
    x_mid = x1 + width//2 if width%2 ==0 else x1 + width//2 + 1
    y_mid = y1 + height//2 if height%2 ==0 else y1 + height//2 + 1
    smaller_dim = min((width, height))
    if smaller_dim < required_size:

        x1 = x_mid - required_size//2
        x2 = x1 + required_size
        y1 = y_mid - required_size//2
        y2 = y1 + required_size

    else:
        if height < width:
            height_new = min((width, pixels_h))
            width_new = min((height_new, width))
        else:
            width_new = min((height, pixels_w))
            height_new = min((width_new, height))
        x1 = x_mid - width_new // 2
        x2 = x1 + width_new

        y1 = y_mid - height_new // 2
        y2 = y1 + height_new

    if x1 < 0:
        remainder = abs(x1)
        x1_new = 0
        x2_new = x2 + remainder
    elif x2 > pixels_w:
        remainder = x2 - pixels_w
        x1_new = x1 - remainder
        x2_new = pixels_w
    else:
        x1_new = x1
        x2_new = x2
    if y1 < 0:
        remainder = abs(y1)
        y1_new = 0
        y2_new = y2 + remainder
    elif y2 > pixels_h:
        remainder = y2 - pixels_h
        y1_new = y1 - remainder
        y2_new = pixels_h
    else:
        y1_new = y1
        y2_new = y2

    face = pixels[y1_new:y2_new, x1_new:x2_new]
    # resize pixels to the model size
    height = y2_new - y1_new
    width = x2_new - x1_new
    image = Image.fromarray(face)
    if height != required_size or width != required_size:
        image = image.resize((required_size, required_size))
    return image

def extract_and_save(img_in, img_out, crop_size):
    pixels = Image.open(img_in)
    height, width = pixels.height, pixels.width
    smaller_dim = min((width,height))
    if smaller_dim < crop_size:
        ratio = smaller_dim / float(crop_size)
        width_new = max((floor(width / ratio), crop_size))
        height_new = max((floor(height / ratio), crop_size))
        pixels = pixels.resize((height_new, width_new))
    pixels = np.asarray(pixels)
    pixels_aligned = extract_face(pixels, crop_size)

    if pixels_aligned is not None:
        if not os.path.exists(os.path.dirname(img_out)):
            os.makedirs(os.path.dirname(img_out))
        pixels_aligned.save(img_out)

def main(dataset_orig, dataset_out, crop_size=224):
    num_files = len(dataset_orig)
    nfiles_processed = 0
    for img_in, img_out in zip(dataset_orig, dataset_out):
        if nfiles_processed % 10000 == 0:
            print(f"Progress {nfiles_processed * 100. / num_files:.3f}% - {nfiles_processed} files out of {num_files} total files")
        nfiles_processed += 1
        if os.path.exists(img_out):
            continue
        try:
            extract_and_save(img_in, img_out, crop_size)
        except (FileNotFoundError, ValueError):
            continue
        # except Exception as e:
        #     print(f"Error processing file {img_in}: {e}.")
    print("Done!")

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    dataset_aligned_output_path = sys.argv[2]
    if not os.path.exists(dataset_aligned_output_path):
        os.makedirs(dataset_aligned_output_path)
    dataset_filelist_path = sys.argv[3]
    with open(dataset_filelist_path, 'r') as f:
        filenames = f.readlines()

    dataset_orig = [os.path.join(dataset_path, fname.strip("\n")) for fname in filenames]
    dataset_aligned = [os.path.join(dataset_aligned_output_path, fname.strip("\n")) for fname in filenames]
    main(dataset_orig, dataset_aligned)
