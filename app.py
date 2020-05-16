import atexit
from flask import Flask, request, send_file, current_app, jsonify
from io import BytesIO
import os
import json
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from keras.backend import set_session
from tensorflow.python.platform import gfile

from project_utilities import extract_face
from project_params import ROOT_DIR
from attacks.blackbox.params import FACENET_WEIGHTS_PATH, \
    NUM_CLASSES_VGGFACE, SQUEEZENET_WEIGHTS_PATH
from attacks.blackbox.adversaries import run_fgsm_attack, \
    run_papernot_attack, generate_adversarial_sample
from attacks.blackbox.substitute_model import load_model
from attacks.whitebox.fgsm.adversary import generate_adv_whitebox

app = Flask(__name__)


ATTACK_TO_FUNC = {
    'fgsm': run_fgsm_attack,
    'papernot': run_papernot_attack,
}

WHITEBOX_KWARGS = {
    'eps': 0.001,
    'num_iter': 500,
}


def load_app_globals():
    app.graph = tf.get_default_graph()
    app.sess = tf.Session(graph=app.graph)

    # Load Facenet model for whitebox
    with app.sess.as_default():
        model_exp = os.path.expanduser(FACENET_WEIGHTS_PATH)
        # print(f'Model filename: {model_exp}')
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='facenet')

    set_session(app.sess)

    # Load MTCNN model
    app.mtcnn = MTCNN()
    app.substitute_model = load_model(os.path.join(ROOT_DIR,
                                                   SQUEEZENET_WEIGHTS_PATH),
                                      NUM_CLASSES_VGGFACE)
    atexit.register(webservice_cleanup, app.sess)


@app.route('/blackbox', methods=['POST'])
def blackbox():
    img = np.array(Image.open(request.files['input']))

    # json_dict = request.json
    attack_name = request.form.get('attack')

    attack_func = None
    if attack_name is not None:
        attack_func = ATTACK_TO_FUNC.get(attack_name)

    attack_args = json.loads(request.form.get('attack_args', '{}'))
    if not attack_func:
        return jsonify({
            'Error': f'Attack type not supported, attack should be one '
                     f'of {set(ATTACK_TO_FUNC.keys())}'
        }), 400

    set_session(current_app.sess)
    face_img = extract_face(current_app.mtcnn, img, (224, 224),
                            graph=current_app.graph).astype(np.float32)

    with current_app.graph.as_default():
        with current_app.sess.as_default():
            adv_img = generate_adversarial_sample(
                face_img, current_app.substitute_model, attack_func,
                attack_args)

    file_object = BytesIO()
    adv_img.save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')


@app.route('/whitebox', methods=['POST'])
def whitebox():
    img = np.array(Image.open(request.files['input']))
    target_img = np.array(Image.open(request.files['target']))

    req_whitebox_kwargs = {}
    for arg, default_val in WHITEBOX_KWARGS.items():
        req_whitebox_kwargs[arg] = request.form.get(arg, default_val)

    set_session(current_app.sess)

    face_img = extract_face(current_app.mtcnn, img, (160, 160),
                            graph=current_app.graph).astype(np.float32)
    target_img = extract_face(current_app.mtcnn, target_img, (160, 160),
                              graph=current_app.graph).astype(np.float32)

    with current_app.graph.as_default():
        adv_img = generate_adv_whitebox(face_img, target_img,
                                        current_app.graph, current_app.sess,
                                        **req_whitebox_kwargs)

    file_object = BytesIO()
    Image.fromarray(adv_img).save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')


@app.route('/face_align', methods=['POST'])
def face_align():
    img = np.array(Image.open(request.files['input']))

    set_session(current_app.sess)

    face_img = extract_face(current_app.mtcnn, img, (160, 160),
                            graph=current_app.graph)

    file_object = BytesIO()
    Image.fromarray(face_img).save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')


def webservice_cleanup(sess):
    try:
        sess.close()
        print("tf session closed successfully")
    except Exception as e:
        print(f"Error when closing tf session: {e}")


if __name__ == '__main__':
    load_app_globals()
    # app.debug = True
    app.run(host='0.0.0.0', threaded=True)
