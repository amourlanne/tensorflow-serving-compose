#requests are objects that flask handles (get set post, etc)
from flask import Flask, request, render_template
# library for reading and resizing images
from PIL import Image
# for matrix math
import numpy as np
# for reading operating system data
import os

import base64

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

app = Flask(__name__)

model = load_model("./models/mnist_cnn.h5")

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():

    img_data = request.get_data().decode('utf-8')
    # separate the metadata from the image data
    head, data = img_data.split(',', 1)
    # save Image
    with open('image.png', 'wb') as f:
            f.write(base64.b64decode(data))

    # read the image and convert to grayscale
    img = Image.open('image.png').convert("L")
    # make it the right size
    img = img.resize(size=(28, 28))
    # transform to numpy array
    img = np.array(img)
    # compute a bit-wise inversion so black becomes white and vice versa
    img = np.invert(img)
    # convert to a 4D tensor to feed into our model
    img = img.reshape(1,28,28,1)

    prediction = np.argmax(model.predict(img), axis=-1)

    return np.array2string(prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT'))