from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import os
import time
from utils.preprocessing import extract_video_features, create_inception_cnn_model, predict_video_class 
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
cnn_model = create_inception_cnn_model()
model = tf.keras.models.load_model('models/LstmModelTraining.h5', compile=False)

@app.route('/')
def index():
    return render_template('index.html')

label_encoder = LabelEncoder()
label_encoder.fit(["distracted behavior", "normal behavior"])

@app.route('/predict', methods=['POST'])
def predict():
    video = request.files['video']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'upload.mp4')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    video.save(filepath)

    # Use the unified prediction function
    predicted_label = predict_video_class(filepath, cnn_model, model, label_encoder)

    if predicted_label is None:
        return render_template('index.html', prediction="No frames extracted or prediction failed")

    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
