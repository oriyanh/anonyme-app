from attacks.blackbox.substitute_model import load_model
from attacks.blackbox.params import SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE, extract_face
from attacks.blackbox.adversaries import generate_adversarial_sample, run_fgsm_attack, run_papernot_attack

import json
import numpy as np
# from tensorflow.python.framework import ops
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from PIL import Image
from mtcnn import MTCNN

app = Flask(__name__)

# TODO: Load model outside of request-response loop

ATTACK_TO_FUNC = {
    'fgsm': run_fgsm_attack,
    'papernot': run_papernot_attack,
}

def load_app_globals():
    app.substitute_model = load_model(SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE)
    app.mtcnn = MTCNN()
    global graph
    graph = tf.get_default_graph()

@app.route('/anonymize', methods=['POST'])
def anonymize():
    img = np.array(Image.open(request.files['input']))
    # json_dict = request.json
    attack = request.form.get('attack')
    attack_args = json.loads(request.form.get('attack_args', '{}'))
    if attack is None or not ATTACK_TO_FUNC.get(attack):
        return jsonify({
            'Error': f'Attack type not supported, attack should be one of {set(ATTACK_TO_FUNC.keys())}'
        }), 400
    with graph.as_default():
        adv_img = generate_adversarial_sample(extract_face(img, (224, 224, 3)),
                                              ATTACK_TO_FUNC[attack], **attack_args)

    file_object = BytesIO()
    adv_img.save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')

if __name__ == '__main__':
    load_app_globals()
    app.run(host='0.0.0.0')
