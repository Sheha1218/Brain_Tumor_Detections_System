from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


def load_trained_model(path="model.h5"):
    try:
        model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded successfully! Expected input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_trained_model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_disease(image_path):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        image = cv2.imread(image_path)
        
        
        input_size = model.input_shape[1:3]  # (height, width)
        image = cv2.resize(image, input_size)

        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        return {
            "class": class_labels[class_index],
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    return render_template('index.html')  # make sure this file is in templates/

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_disease(filepath)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
