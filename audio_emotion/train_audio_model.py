import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


DATASET_PATH = "datasets/audio_emotion/archive (16)"
MODEL_DIR = "audio_emotion/model"

MODEL_PATH = os.path.join(MODEL_DIR, "audio_emotion_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "audio_label_encoder.pkl")

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


def train_audio_model():
    X = []
    y = []

    print("Loading RAVDESS audio dataset...")

    for actor_folder in os.listdir(DATASET_PATH):
        actor_path = os.path.join(DATASET_PATH, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        for file_name in os.listdir(actor_path):
            if file_name.endswith(".wav"):
                emotion_code = file_name.split("-")[2]
                emotion = emotion_map.get(emotion_code)

                if emotion is None:
                    continue

                file_path = os.path.join(actor_path, file_name)

                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion)
                except Exception as e:
                    print("Error:", file_path, e)

    X = np.array(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("Training audio emotion model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("\nAudio emotion model saved successfully.")


if __name__ == "__main__":
    train_audio_model()