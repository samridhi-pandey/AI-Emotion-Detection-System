import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "face_emotion_model.h5"

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def predict_face_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        prediction = model.predict(face, verbose=0)

        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = float(np.max(prediction))

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "box": (x, y, w, h)
        }

    return None