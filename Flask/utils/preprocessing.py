import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# 1. Define CNN model using InceptionV3
def create_inception_cnn_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# 2. Initialize CNN model once (global)
cnn_model = create_inception_cnn_model()

def extract_video_features(video_path, feature_extractor, max_sequence_length=50, frame_step=5, target_size=(299, 299)):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_step == 0:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_input(frame.astype(np.float32))
            frames.append(frame)
        frame_id += 1

    cap.release()

    if not frames or len(frames) < 10:
        print("Warning: No frames or too few extracted.")
        return None

    frames = np.array(frames, dtype=np.float32)
    features = feature_extractor.predict(frames, verbose=0)

    features_padded = pad_sequences(
        [features], maxlen=max_sequence_length,
        dtype='float32', padding='post', truncating='post'
    )[0]

    return features_padded

label_encoder = LabelEncoder()
label_encoder.fit(["distracted behavior", "normal behavior"])  

def predict_video_class(video_path, feature_extractor, model, label_encoder, max_sequence_length=50):
    features = extract_video_features(video_path, feature_extractor)
    if features is None:
        print("Feature extraction failed.")
        return None

    features = np.expand_dims(features, axis=0)  

    prediction = model.predict(features, verbose=0)
    print("Prediction vector:", prediction)

    predicted_class = int(prediction[0][0] > 0.5)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    print(f"Predicted class: {predicted_label}")
    return predicted_label