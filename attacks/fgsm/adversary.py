import tensorflow as tf
from tensorflow.python.platform import gfile
from imageio import imread, imwrite
import os
import numpy as np

OUTPUT_PATH = '/cs/ep/503/outputs/fgsm'
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

    imwrite(os.path.join(OUTPUT_PATH, file_name), modified_img)


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
    def __init__(self, model):
        self.sess = tf.Session()
        with self.sess.as_default():
            model_exp = os.path.expanduser(model)
            print(f'Model filename: {model_exp}')
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

    def generate_embedding(self, input_img):
        regularized_img = regularize_img(input_img)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [regularized_img], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def compare(self, input_pic, target_pic):
        embedding1 = self.generate_embedding(input_pic)
        embedding2 = self.generate_embedding(target_pic)

        # if both pictures are same, return 0
        return euclidian_distance(embedding1, embedding2)

    def generate_adv_whitebox(self, input_path, target_path, eps, num_iter):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        target_emb = self.generate_embedding(target_path)

        # Calculates loss
        loss = tf.sqrt(tf.reduce_sum(tf.square(embeddings - target_emb)))

        # Run fgsm
        adversary = fgsm(images_placeholder, loss=loss, eps=eps, bounds=(-1.0, 1.0))

        input_image = regularize_img(input_path)
        adv_img = input_image.reshape(-1, *input_image.shape)

        last_adv_loss = 0
        cnt = 0
        within_loss_limit = False

        for i in range(num_iter):
            feed_dict = {
                images_placeholder: adv_img,
                phase_train_placeholder: False
            }
            adv_img, adv_loss = self.sess.run([adversary, loss], feed_dict=feed_dict)

            # img_bias = np.sqrt(np.sum(np.square(adv_img[0, ...] - input_image)))
            img_bias = euclidian_distance(adv_img[0, ...], input_image)

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

        save_img(adv_img[0, ...], input_path, target_path, eps, iter)

        feed_dict = {
            images_placeholder: adv_img,
            phase_train_placeholder: False
        }

        adv_embedding = self.sess.run(embeddings, feed_dict=feed_dict)[0]
        embedding_dist = euclidian_distance(adv_embedding, target_emb)

        print(f'The distance between input embedding and target is {embedding_dist:2.6f}')


def fgsm(x, loss, eps=0.3, bounds=(0, 1)):
    """

    :param x:
    :param loss:
    :param eps:
    :param bounds:
    :return:
    """

    (clip_min, clip_max) = bounds

    grad, = tf.gradients(loss, x)

    normalized_grad = tf.sign(grad)
    normalized_grad = tf.stop_gradient(normalized_grad)

    scaled_grad = eps * normalized_grad

    adv_x = x - scaled_grad

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


if __name__ == '__main__':

    fr = Adversary('/cs/ep/503/facenet_model.pb')

    input_pic = "/cs/ep/503/amit/fgsm/Bill_Gates_0001.png"
    target_pic = "/cs/ep/503/amit/fgsm/chaoren.png"
    # print fr.compare(input_pic,target_pic)

    iters = [500, 1000, 1500, 2000, 2500]
    eps = [0.001, 0.0025, 0.005, 0.01, 0.02]

    combs = np.transpose([np.tile(iters, len(eps)), np.repeat(eps, len(iters))])

    for [iter, eps] in combs:
        print(f"Now using iter={iter} and eps={eps}")
        fr.generate_adv_whitebox(input_pic, target_pic, eps, int(iter))
        break
    # fr.generate_adv_whitebox(input_pic, target_pic, eps, iter)
