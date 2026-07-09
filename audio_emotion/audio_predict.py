import os
import librosa
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "audio_emotion_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "audio_label_encoder.pkl"




def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)


def predict_audio_mood(audio_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Audio model not found. Train it first.")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    features = extract_features(audio_path)

    prediction = model.predict(features)[0]
    emotion = encoder.inverse_transform([prediction])[0]

    return {
        "detected_emotion": emotion
    }