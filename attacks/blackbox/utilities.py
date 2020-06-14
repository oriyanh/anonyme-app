print(f"Loading module {__file__}")

import os
os.umask(2)
import numpy as np
import tensorflow as tf
from PIL import Image
from shutil import rmtree


def extract_face(mtcnn, pixels, bounding_box_size=256, crop_size=224,
                 graph=tf.get_default_graph()):
    # detect faces in the image
    pixels_h, pixels_w, _ = pixels.shape
    with graph.as_default():
        results = mtcnn.detect_faces(pixels)
        if not results:  # No bounding box
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


def standardize_batch(batch, center=True):
    """

    :param batch:  Numpy array, dimensions (N, 224, 224, 3) , channels in RGB
    :param center: boolean, if True will remove mean from batch, if False will add mean to batch.
    :return: standardized batch
    """
    x_temp = np.copy(batch).astype(np.float32)
    if center:
        x_temp[..., 0] -= 131.0912
        x_temp[..., 1] -= 103.8827
        x_temp[..., 2] -= 91.4953
    else:
        x_temp[..., 0] += 131.0912
        x_temp[..., 1] += 103.8827
        x_temp[..., 2] += 91.4953

    return x_temp


def get_train_set(train_dir, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                              horizontal_flip=True,
                                                              zoom_range=0.15)
    train_it = datagen.flow_from_directory(train_dir, class_mode='sparse', batch_size=batch_size,
                                           shuffle=True, target_size=(224, 224))
    nbatches = train_it.n // batch_size
    if nbatches * batch_size < train_it.n:
        nbatches += 1

    def gen():
        while True:
            x_train, y_train = train_it.next()
            yield x_train.astype(np.float32) / 255.0, y_train.astype(np.int)

    return gen(), nbatches, train_it.num_classes, train_it.class_indices


def get_validation_set(validation_dir, train_class_indices, batch_size):
    # TODO validation dir is assumed to be mapped properly, need to resolve this
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    validation_it = datagen.flow_from_directory(validation_dir, class_mode='sparse',
                                                batch_size=batch_size,
                                                shuffle=True, target_size=(224, 224))
    class_indices = validation_it.class_indices
    class_map = {idx: name for name, idx in class_indices.items()}
    nbatches = validation_it.n // batch_size
    if nbatches * batch_size < validation_it.n:
        nbatches += 1

    def gen():
        while True:
            x_val, y_unmapped = validation_it.next()

            y_mapped = [class_map[y] for y in y_unmapped]
            y_val = np.array([train_class_indices.get(y, -1) for y in y_mapped])
            print(f"Number of predictions from non existent classes: {np.count_nonzero(y_val < 0)}")
            y_val = np.array([train_class_indices.get(y, 194) for y in y_mapped])
            yield x_val.astype(np.float32) / 255.0, y_val.astype(np.int)

    return gen(), nbatches

def oracle_classify_and_save(oracle, dataset_dir, output_dir, batch_size, prune_threshold=0):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_it = datagen.flow_from_directory(dataset_dir, class_mode=None, batch_size=batch_size,
                                           shuffle=True, target_size=(224, 224))

    nbatches = train_it.n // batch_size
    if nbatches * batch_size < train_it.n:
        nbatches += 1

    with graph.as_default():
        tf.keras.backend.set_session(sess)
        for step in range(nbatches):
            print(f"Progress {(step+1)*100/nbatches:.3f}%")
            images = train_it.next()
            label_batch = oracle.predict(images)
            labels = np.argmax(label_batch, axis=1)
            save_batch(images, labels, output_dir)
    if prune_threshold > 0:
        prune_dataset(output_dir, prune_threshold)

def save_batch(images, labels, output_dir):
    for image, label in zip(images, labels):
        dir_name = "n" + f'{label}'.rjust(6, '0')
        label_dir = os.path.join(output_dir, dir_name)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        fname = f"{np.random.randint(9)}".rjust(9, '0') + ".jpg"
        output = os.path.join(label_dir, fname)
        while os.path.exists(output):
            fname = f"{np.random.randint(100000000)}".rjust(9, '0') + ".jpg"
            output = os.path.join(label_dir, fname)

        img = Image.fromarray(image.astype(np.uint8))
        img.save(output)

def prune_dataset(dataset_dir, threshold=10):
    print(f"Pruning classes from {dataset_dir} that have less than {threshold} samples")
    sub_directories = os.listdir(dataset_dir)
    print(f"Initial number of classes: {len(sub_directories)}")
    for sub_dir in sub_directories:
        class_path = os.path.join(dataset_dir, sub_dir)
        num_samples = len(os.listdir(class_path))
        if num_samples < threshold:
            print(f"Pruning class {sub_dir}")
            rmtree(class_path, ignore_errors=True)
    num_classes = len(os.listdir(dataset_dir))
    print(f"Remaining number of classes: {num_classes}")

class Singleton(type):
    """
    Metaclass to be used for singleton (used for representing blackbox model
    class)
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]

# config = tf.ConfigProto(device_count={'GPU': 0})  # Used on Oriyan's computer :)
config = tf.ConfigProto()
graph = tf.get_default_graph()
sess = tf.Session(graph=graph, config=config)

print(f"Successfully loaded module {__file__}")