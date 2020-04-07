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

from attacks.blackbox.params import extract_face, FACENET_WEIGHTS_PATH, \
    SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE
from attacks.blackbox.adversaries import run_fgsm_attack, \
    run_papernot_attack, generate_adversarial_sample
from attacks.blackbox.substitute_model import load_model
from attacks.fgsm.adversary import generate_adv_whitebox

app = Flask(__name__)


ATTACK_TO_FUNC = {
    'fgsm': run_fgsm_attack,
    'papernot': run_papernot_attack,
}

WHITEBOX_KWARGS = {'eps', 'num_iter'}


def load_app_globals():
    app.graph = tf.get_default_graph()
    app.substitute_model = load_model(SUBSTITUTE_WEIGHTS_PATH,
                                      NUM_CLASSES_VGGFACE)

    app.sess = tf.Session(graph=app.graph)

    # Load Facenet model for whitebox
    with app.sess.as_default():
        model_exp = os.path.expanduser(FACENET_WEIGHTS_PATH)
        # print(f'Model filename: {model_exp}')
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='facenet')
        # app.sess.run(tf.global_variables_initializer())

    set_session(app.sess)

    # Load MTCNN model
    app.mtcnn = MTCNN()


@app.route('/blackbox', methods=['POST'])
def blackbox():
    img = np.array(Image.open(request.files['input']))

    # json_dict = request.json
    attack = request.form.get('attack')
    attack_args = json.loads(request.form.get('attack_args', '{}'))
    if attack is None or not ATTACK_TO_FUNC.get(attack):
        return jsonify({
            'Error': f'Attack type not supported, attack should be one '
                     f'of {set(ATTACK_TO_FUNC.keys())}'
        }), 400

    set_session(current_app.sess)
    face_img = extract_face(current_app.mtcnn, img, (224, 224),
                            graph=current_app.graph).astype(np.float32)

    with current_app.sess.as_default():
        with current_app.graph.as_default():
            adv_img = generate_adversarial_sample(
                face_img, ATTACK_TO_FUNC[attack], attack_args)

    file_object = BytesIO()
    adv_img.save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')


@app.route('/whitebox', methods=['POST'])
def whitebox():
    img = np.array(Image.open(request.files['input']))
    target_img = np.array(Image.open(request.files['target']))

    req_whitebox_kwargs = {}
    for arg in WHITEBOX_KWARGS:
        if arg in request.form:
            req_whitebox_kwargs[arg] = request.form[arg]

    set_session(current_app.sess)

    face_img = extract_face(current_app.mtcnn, img, (160, 160),
                            graph=current_app.graph).astype(np.float32)
    target_img = extract_face(current_app.mtcnn, target_img, (160, 160),
                              graph=current_app.graph).astype(np.float32)

    with current_app.graph.as_default():
        adv_img = generate_adv_whitebox(face_img, target_img,
                                        **req_whitebox_kwargs)

    file_object = BytesIO()
    Image.fromarray(adv_img).save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')


@app.route('/face_align', methods=['POST'])
def face_align():
    img = np.array(Image.open(request.files['input']))
    face_img = extract_face(img, (160, 160))

    file_object = BytesIO()
    Image.fromarray(face_img).save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')


def webservice_cleanup():
    try:
        current_app.sess.close()
        print("tf session closed successfully")
    except Exception as e:
        print(f"Error when closing tf session: {e}")


if __name__ == '__main__':
    load_app_globals()
    atexit.register(webservice_cleanup)
    # app.debug = True
    app.run(host='0.0.0.0', threaded=True)
