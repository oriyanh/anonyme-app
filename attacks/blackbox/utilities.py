import os
import numpy as np
from PIL import Image
import tensorflow as tf

from attacks.blackbox.params import DATASET_TRAIN_LIST, TRAIN_SET_ALIGNED, NUM_CLASSES_VGGFACE


def load_initial_set(num_samples):
    pass

def load_training_set():
    images = []
    with open(DATASET_TRAIN_LIST, 'r') as f:
        for r in f.readlines()[:1000]:
            img_path = os.path.join(TRAIN_SET_ALIGNED, r.strip('\n\r'))
            try:
                img = Image.open(img_path)
                img_np = np.asarray(img)
                # face = extract_face(img_np)
                # images.append(face)
                images.append(img_np)
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")
    return images

def load_test_set():
    pass

def extract_face(mtcnn, pixels, required_size=(224, 224),
                 graph=tf.get_default_graph()):
    # detect faces in the image
    with graph.as_default():
        results = mtcnn.detect_faces(pixels)

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def get_training_set(train_dir, batch_size):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_it = datagen.flow_from_directory(train_dir, class_mode='sparse', batch_size=batch_size,
                                           shuffle=True, target_size=(224, 224))
    nbatches = train_it.n // batch_size
    if nbatches * batch_size < train_it.n:
        nbatches += 1

    def gen():
        while True:
            x_train, y_train = train_it.next()
            yield x_train.astype(np.float32) / 255.0, y_train.astype(np.int)

    # ds_images = tf.data.Dataset.from_generator(gen, output_shapes=([None, 224, 224, 3], [None]),
    #                                            output_types=(tf.float32, tf.int32))
    # return ds_images, train_it.n
    return gen(), nbatches, train_it.num_classes

def predict_and_save(oracle, dataset_dir, output_dir, batch_size):
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

config = tf.ConfigProto(device_count={'GPU': 0})
graph = tf.get_default_graph()
sess = tf.Session(graph=graph, config=config)