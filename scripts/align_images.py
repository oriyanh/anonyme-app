import sys
import os
import numpy as np
from mtcnn import MTCNN
from PIL import Image

detector = MTCNN()

def extract_face(pixels, bounding_box_size=256, crop_size=224):
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

    if pixels_w < bounding_box_size or pixels_h < bounding_box_size:
        if pixels_w < bounding_box_size:
            x1 = 0
            x2 = pixels_w
        else:
            x2 = x1 + width
        if pixels_h < bounding_box_size:
            y1 = 0
            y2 = pixels_h
        else:
            y2 = y1 + height

    elif smaller_dim < bounding_box_size:

        x1 = x_mid - bounding_box_size // 2
        x2 = x1 + bounding_box_size
        y1 = y_mid - bounding_box_size // 2
        y2 = y1 + bounding_box_size

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

    if (y2_new - y1_new) >= pixels_h:
        y1_new = 0
        y2_new = pixels_h
    if (x2_new - x1_new) >= pixels_w:
        x1_new = 0
        x2_new = pixels_w

    x1 = np.random.randint(x1_new, x2_new-crop_size)
    x2 = x1 + crop_size
    y1 = np.random.randint(y1_new, y2_new-crop_size)
    y2 = y1 + crop_size
    face = pixels[y1:y2, x1:x2]
    return face

def extract_and_save(source_path, target_path, bounding_box_size, crop_size):
    image_in = Image.open(source_path)
    height, width = image_in.height, image_in.width
    if height < crop_size or width < crop_size:
        return

    image = np.asarray(image_in)
    image_aligned = extract_face(image, bounding_box_size, crop_size)

    if image_aligned is not None:
        image_out = Image.fromarray(image_aligned)
        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path))
        image_out.save(target_path)

def main(dataset_orig, dataset_out, bounding_box=256, crop_size=224):
    num_files = len(dataset_orig)
    nfiles_processed = 0
    for img_in, img_out in zip(dataset_orig, dataset_out):
        if nfiles_processed % 10000 == 0:
            print(f"Progress {nfiles_processed * 100. / num_files:.3f}% - {nfiles_processed} files out of {num_files} total files")
        nfiles_processed += 1
        if os.path.exists(img_out):
            continue
        try:
            extract_and_save(img_in, img_out, bounding_box, crop_size)
        except (FileNotFoundError, ValueError):
            continue
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
    main(dataset_orig, dataset_aligned, bounding_box=256, crop_size=224)
