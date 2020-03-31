import tensorflow as tf
from flask import current_app
from imageio import imread, imwrite
import cv2
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

    file_name = f"{shortname_input}_{shortname_target}_eps_{eps}_iter_" \
                f"{iter}.png"
    modified_img = modified_img.reshape(*modified_img.shape[:3])

    imwrite(os.path.join(OUTPUT_PATH, file_name), modified_img)


def regularize_img(input_img):
    if type(input_img) is str:
        img = imread(os.path.expanduser(input_img), as_gray=False,
                     pilmode='RGB')
    else:
        img = input_img.copy()
    regularized_img = (img * 2.0 / 255.0) - 1.0
    return regularized_img


def euclidian_distance(embeddings1, embeddings2):
    # Euclidian distance
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff))
    return np.sqrt(dist)


def generate_embedding(input_img):
    regularized_img = regularize_img(input_img)

    # Get input and output tensors
    images_placeholder = current_app.graph.get_tensor_by_name(
        "facenet/input:0")
    embeddings = current_app.graph.get_tensor_by_name("facenet/embeddings:0")
    phase_train_placeholder = current_app.graph.get_tensor_by_name(
        "facenet/phase_train:0")

    # Run forward pass to calculate embeddings
    feed_dict = {
        images_placeholder: [regularized_img],
        phase_train_placeholder: False,
    }
    return current_app.sess.run(embeddings, feed_dict=feed_dict)[0]


def generate_adv_whitebox(input_path, target_path, eps=0.001, num_iter=500):
    # Get input and output tensors
    images_placeholder = current_app.graph.get_tensor_by_name(
        "facenet/input:0")
    embeddings = current_app.graph.get_tensor_by_name("facenet/embeddings:0")
    phase_train_placeholder = current_app.graph.get_tensor_by_name(
        "facenet/phase_train:0")

    target_emb = generate_embedding(target_path)

    # Calculates loss
    loss = tf.sqrt(tf.reduce_sum(tf.square(embeddings - target_emb)))

    # Run fgsm
    adversary = fgsm(images_placeholder, loss=loss, eps=eps,
                     bounds=(-1.0, 1.0))

    input_image = regularize_img(input_path)
    target_img = regularize_img(target_path)
    adv_img = input_image.reshape(-1, *input_image.shape)

    last_adv_loss = 0
    cnt = 0
    within_loss_limit = False

    for i in range(num_iter):
        feed_dict = {
            images_placeholder: adv_img,
            phase_train_placeholder: False
        }
        adv_img, adv_loss = current_app.sess.run([adversary, loss],
                                                 feed_dict=feed_dict)

        # img_bias = np.sqrt(np.sum(np.square(adv_img[0, ...] - input_image)))
        img_bias = euclidian_distance(adv_img[0, ...], input_image)
        target_bias = euclidian_distance(adv_img[0, ...], target_img)
        print(f'Iteration #{i + 1}/{num_iter}\n'
              f'Bias from original image: {img_bias:2.6f}\n'
              f'Bias from target image: {target_bias:2.6f}\n'
              f'L2 Loss from target: {adv_loss:2.6f}\n\n')
        if img_bias > target_bias:
            print('Image embedding is now closer to target embedding')
            break

        if np.absolute(adv_loss - last_adv_loss) < LOSS_LIMIT:
            if within_loss_limit:
                cnt += 1
            else:
                within_loss_limit = True
        else:
            within_loss_limit = False
            cnt = 0

        if cnt == LOSS_CNT_THRESHOLD:
            print('Loss convergence reached')
            break

        if adv_loss < ADV_LOSS_STOP:
            print('Loss threshold reached')
            break

        last_adv_loss = adv_loss

    # feed_dict = {
    #     images_placeholder: adv_img,
    #     phase_train_placeholder: False
    # }
    #
    # adv_embedding = self.sess.run(embeddings, feed_dict=feed_dict)[0]
    # embedding_dist = euclidian_distance(adv_embedding, target_emb)
    #
    # print(f'The distance between input embedding and target is {embedding_dist:2.6f}')

    # adv_img = adv_img[0, ...].reshape(*adv_img[0, ...].shape[:3])
    adv_img = cv2.normalize(adv_img[0, ...], None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return adv_img.astype(np.uint8)


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

    # fr = Adversary('/cs/ep/503/facenet_model.pb')

    input_pic = "/cs/ep/503/inputs/Kofi_Annan_0003.png"
    target_pic = "/cs/ep/503/amit/facenet/data/images/test_aligned/Silvio_Berlusconi/Silvio_Berlusconi_0017.png"
    # print fr.compare(input_pic,target_pic)

    iters = [500, 1000, 1500, 2000, 2500]
    eps = [0.001, 0.0025, 0.005, 0.01, 0.02]

    combs = np.transpose([np.tile(iters, len(eps)), np.repeat(eps, len(iters))])

    for [iter, eps] in combs:
        print(f"Now using iter={iter} and eps={eps}")
        # fr.generate_adv_whitebox(input_pic, target_pic, eps, int(iter))
        # break
    # fr.generate_adv_whitebox(input_pic, target_pic, eps, iter)
