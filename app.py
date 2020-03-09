from io import BytesIO

from flask import Flask, request, send_file
from PIL import Image
from attacks.blackbox.train import run_blackbox
import numpy as np

app = Flask(__name__)

# TODO: Load model outside of request-response loop

@app.route('/anonymize', methods=['POST'])
def anonymize():
    img = Image.open(request.files['input'])
    # with graph.as_default():
    run_blackbox(np.asarray(img))
    file_object = BytesIO()
    img.save(file_object, 'jpeg')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')

app.run(host='0.0.0.0')