import tensorflow as tf
from io import BytesIO
from attacks.blackbox.substitute_model import load_model
from attacks.blackbox.params import SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE
from flask import Flask, request, send_file
from PIL import Image
from mtcnn import MTCNN
app = Flask(__name__)

# TODO: Load model outside of request-response loop

def load():
    app.substitute_model = load_model(SUBSTITUTE_WEIGHTS_PATH, NUM_CLASSES_VGGFACE)
    app.mtcnn = MTCNN()
    global graph
    graph = tf.get_default_draph()

@app.route('/anonymize', methods=['POST'])
def anonymize():
    img = Image.open(request.files['input'])
    # with graph.as_default():

    file_object = BytesIO()
    img.save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')

if __name__ == '__main__':
    load()
    app.run(host='0.0.0.0')
