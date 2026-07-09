import joblib
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "goemotions_text_model.pkl"
BINARIZER_PATH = BASE_DIR / "model" / "label_binarizer.pkl"
EMOTIONS_PATH = BASE_DIR.parent / "datasets" / "text_emotion" / "goemotions" / "emotions.txt"


model = joblib.load(MODEL_PATH)
mlb = joblib.load(BINARIZER_PATH)


with open(EMOTIONS_PATH, "r", encoding="utf-8") as file:
    emotions = [line.strip() for line in file.readlines()]


def predict_emotions(text, top_k=3):
    probabilities = model.predict_proba([text])

    probs = probabilities[0]

    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "emotion": emotions[idx],
            "confidence": round(float(probs[idx]), 4)
        })

    return results


if __name__ == "__main__":
    text = input("Enter text: ")

    predictions = predict_emotions(text)

    print("\nTop Predicted Emotions:\n")

    for pred in predictions:
        print(
            f"{pred['emotion']} -> {pred['confidence']}"
        )