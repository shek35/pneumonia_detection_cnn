import cv2
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('CNN_model.h5')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image):
    # Convert the PIL Image object to a NumPy array
    img = np.array(image)

    # Resize the image
    img = cv2.resize(img, (128, 128))

    # Convert grayscale to RGB if necessary
    if len(img.shape) == 2:  # If the image is grayscale
        img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB

    # Normalize and reshape the image
    img = img.reshape(1, 128, 128, 3)
    img = img / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image = Image.open(file_path)
            image = preprocess_image(image)
            prediction = model.predict(image)
            result = 'Pneumonia' if np.argmax(prediction) == 1 else 'Normal'
            return render_template('index.html', prediction=result, img_path=file_path)
    return render_template('index.html', prediction=None, img_path=None)

if __name__ == "__main__":
    app.run(debug=True)
