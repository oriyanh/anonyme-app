import sys
import os
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from PIL import Image


detector = MTCNN()

def extract_face(pixels, required_size=(224, 224)):
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return image

def extract_and_save(img_in, img_out, crop_size):
    print(f"Aligning {img_in}")
    pixels = plt.imread(img_in)
    pixels_aligned = extract_face(pixels, crop_size)

    if not os.path.exists(os.path.dirname(img_out)):
        os.makedirs(os.path.dirname(img_out))
    pixels_aligned.save(img_out)
    print(f"Saved aligned image to {img_out}")

def main(dataset_orig, dataset_out, crop_size=(224, 224)):
    for img_in, img_out in zip(dataset_orig, dataset_out):
        try:
            extract_and_save(img_in, img_out, crop_size)
        except Exception as e:
            print(f"Error processing file {img_in}: Exception: [ {e} ]", file=sys.stderr)

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
