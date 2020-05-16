import math
import pickle

import tensorflow as tf
from scipy.misc import imread, imsave
import os
import numpy as np
import facenet
OUTPUT_PATH = '/cs/ep/503/outputs/normal_noise'
ADV_LOSS_STOP = 0.01
LOSS_LIMIT = 0.0008
LOSS_CNT_THRESHOLD = 10

def save_img(modified_img, input_path, target_path, eps, iter):
    (filepath, temp_input) = os.path.split(input_path)
    (shortname_input, extension) = os.path.splitext(temp_input)

    (filepath, temp_target) = os.path.split(target_path)
    (shortname_target, extension) = os.path.splitext(temp_target)

    file_name = f"{shortname_input}_{shortname_target}_eps_{eps}_iter_{iter}.png"
    modified_img = modified_img.reshape(*modified_img.shape[:3])
    # imwrite(os.path.join(OUTPUT_PATH, file_name), modified_img)
    imsave(os.path.join(OUTPUT_PATH, file_name), modified_img)


def regularize_img(input_img):
    if type(input_img) is str:
        img = imread(os.path.expanduser(input_img), as_gray=False, pilmode='RGB')
    else:
        img = input_img.copy()
    regularized_img = img * 2.0 / 255.0 - 1.0
    return regularized_img


def euclidian_distance(embeddings1, embeddings2):
    # Euclidian distance
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff))
    return np.sqrt(dist)


class Adversary:
    # def __init__(self, model):
    #     self.sess = tf.Session()
    #     with self.sess.as_default():
    #         model_exp = os.path.expanduser(model)
    #         print(f'Model filename: {model_exp}')
    #         with gfile.FastGFile(model_exp, 'rb') as f:
    #             graph_def = tf.GraphDef()
    #             graph_def.ParseFromString(f.read())
    #             tf.import_graph_def(graph_def, name='')

    def __init__(self, model_path, classifier_pkl, data_dir, image_size=160, batch_size=90, seed=666):
        self.facenet_model = model_path
        self.image_size = image_size
        self.batch_size = batch_size
        np.random.seed(seed=seed)
        dataset = facenet.get_dataset(data_dir)

        # Check that there are at least one training image per class
        for cls in dataset:
            assert (
                len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

        self.paths, self.labels = facenet.get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(self.paths))

        # Load the model
        with open(classifier_pkl, 'rb') as classifier_f:
            self.classifier, self.class_names = pickle.load(classifier_f)

        print('Loaded classifier model from file "%s"' % classifier_pkl)

    def classify(self, paths):
        with tf.Session() as sess:
            print('Loading feature extraction model')
            facenet.load_model(self.facenet_model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * self.batch_size
                end_index = min((i + 1) * self.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, self.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            predictions = self.classifier.predict_proba(emb_array)
            return predictions

    def generate_embedding(self, input_img):
        regularized_img = regularize_img(input_img)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [regularized_img], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


    @tf.function
    def classify2(self, emb):
        return self.classifier.predict_proba(emb)

    def generate_adv_whitebox(self, input_path, target_path, eps, num_iter):
        input_img = facenet.load_data([input_path], False, False, self.image_size)
        target_img = facenet.load_data([target_path], False, False, self.image_size)
        adv_img = input_img.copy()
        target_logits = self.classify([target_path])
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        noise = tf.get_variable("Noise", shape=input_img.shape, trainable=True, initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
        adversary = images_placeholder + noise
        target_label = np.argmax(target_logits, axis=1)
        embeddings_ph = tf.placeholder(tf.float32, shape=embeddings.shape)
        logits = self.classify2(embeddings)

        # Calculates loss
        loss = tf.losses.sparse_softmax_cross_entropy(target_label, logits)
        optimizer = tf.train.AdamOptimizer(eps)
        training_step = optimizer.minimize(loss)
        with tf.Session() as sess:
            last_adv_loss = 0
            cnt = 0
            within_loss_limit = False
            for i in range(num_iter):
                feed_dict = {
                    images_placeholder: input_img,
                    phase_train_placeholder: False
                }
                _, noise, adv_loss = sess.run([training_step, noise, loss], feed_dict=feed_dict)
                adv_img = input_img + noise
                img_bias = euclidian_distance(adv_img, input_img)

                print(f'{i} Bias from original image: {img_bias:2.6f} Loss: {adv_loss:2.6f}')
                if np.absolute(adv_loss - last_adv_loss) < LOSS_LIMIT:
                    if within_loss_limit:
                        cnt += 1
                    else:
                        within_loss_limit = True
                else:
                    within_loss_limit = False
                    cnt = 0

                if cnt == LOSS_CNT_THRESHOLD:
                    print('Convergence reached')
                    break

                if adv_loss < ADV_LOSS_STOP:
                    print('Loss threshold reached')
                    break

                last_adv_loss = adv_loss

            save_img(adv_img, input_path, target_path, eps, iter)

            feed_dict = {
                images_placeholder: adv_img,
                phase_train_placeholder: False
            }

            adv_embedding = sess.run(embeddings, feed_dict=feed_dict)[0]
            feed_dict = {
                images_placeholder: target_img,
                phase_train_placeholder: False
            }
            target_emb = sess.run(embeddings, feed_dict=feed_dict)[0]
            embedding_dist = euclidian_distance(adv_embedding, target_emb)

            print(f'The distance between input embedding and target is {embedding_dist:2.6f}')
        # Run fgsm
        # adversary = trainable_noise(images_placeholder, noise)
        #
        # input_image = regularize_img(input_path)
        # adv_img = input_image.reshape(-1, *input_image.shape)

        # last_adv_loss = 0
        # cnt = 0
        # within_loss_limit = False
        #
        # for i in range(num_iter):
        #     feed_dict = {
        #         images_placeholder: adv_img,
        #         phase_train_placeholder: False
        #     }
        #     adv_img, adv_loss = self.sess.run([training_step, adversary, loss], feed_dict=feed_dict)
        #
        #     # img_bias = np.sqrt(np.sum(np.square(adv_img[0, ...] - input_image)))
        #     img_bias = euclidian_distance(adv_img[0, ...], input_image)
        #
        #     print(f'{i} Bias from original image: {img_bias:2.6f} Loss: {adv_loss:2.6f}')
        #
        #     if np.absolute(adv_loss - last_adv_loss) < LOSS_LIMIT:
        #         if within_loss_limit:
        #             cnt += 1
        #         else:
        #             within_loss_limit = True
        #     else:
        #         within_loss_limit = False
        #         cnt = 0
        #
        #     if cnt == LOSS_CNT_THRESHOLD:
        #         print('Convergence reached')
        #         break
        #
        #     if adv_loss < ADV_LOSS_STOP:
        #         print('Loss threshold reached')
        #         break
        #
        #     last_adv_loss = adv_loss
        #
        # save_img(adv_img[0, ...], input_path, target_path, eps, iter)
        #
        # feed_dict = {
        #     images_placeholder: adv_img,
        #     phase_train_placeholder: False
        # }
        #
        # adv_embedding = self.sess.run(embeddings, feed_dict=feed_dict)[0]
        # embedding_dist = euclidian_distance(adv_embedding, target_emb)
        #
        # print(f'The distance between input embedding and target is {embedding_dist:2.6f}')


if __name__ == '__main__':

    facenet_model = '/cs/ep/503/facenet/facenet/models/20180402-114759.pb'
    classifier = '/cs/ep/503/facenet/facenet/models/my_classifier.pkl'
    test_data = '/cs/ep/503/facenet/facenet/data/images/test_aligned'
    fr = Adversary(facenet_model, classifier, test_data)

    input_pic = "/cs/ep/503/facenet/facenet/data/images/test_aligned/Guillermo_Coria/Guillermo_Coria_0001.png"
    target_pic = "/cs/ep/503/facenet/facenet/data/images/test_aligned/Silvio_Berlusconi/Silvio_Berlusconi_0017.png"
    # print fr.compare(input_pic,target_pic)
    fr.generate_adv_whitebox(input_pic, target_pic, 0.001, 2000)
    # iters = [500, 1000, 1500, 2000, 2500]
    # eps = [0.001, 0.0025, 0.005, 0.01, 0.02]
    #
    # combs = np.transpose([np.tile(iters, len(eps)), np.repeat(eps, len(iters))])
    #
    # for [iter, eps] in combs:
    #     print(f"Now using iter={iter} and eps={eps}")
    #     fr.generate_adv_whitebox(input_pic, target_pic, eps, int(iter))
        # break
    # fr.generate_adv_whitebox(input_pic, target_pic, eps, iter)