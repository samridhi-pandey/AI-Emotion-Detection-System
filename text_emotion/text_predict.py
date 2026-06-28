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

def rule_based_emotion(text):
    text = text.lower()
    positive_phrases = [
        "going to a trip", "going on a trip", "trip tomorrow",
        "vacation", "holiday", "travel", "picnic"
    ]

    workload_phrases = [
        "lot of work", "a lot of work", "too much work",
        "many tasks", "pending work", "work pressure",
        "under pressure", "overloaded", "busy"
    ]

    if any(phrase in text for phrase in positive_phrases):
        return [
            {"emotion": "excitement", "confidence": 0.90},
            {"emotion": "joy", "confidence": 0.75},
            {"emotion": "curiosity", "confidence": 0.50}
        ]

    if any(phrase in text for phrase in workload_phrases):
        return [
            {"emotion": "nervousness", "confidence": 0.85},
            {"emotion": "fear", "confidence": 0.65},
            {"emotion": "neutral", "confidence": 0.45}
        ]
    stress_phrases = [
        "lot of work", "a lot of work", "too much work",
        "many tasks", "pending work", "work pressure",
        "under pressure", "overloaded", "busy"
    ]

    stress_words = [
        "exam", "test", "viva", "assignment", "deadline",
        "marks", "result", "fail", "interview", "presentation",
        "nervous", "scared", "worried", "tension", "stress",
        "work", "workload", "busy", "pressure", "overload",
        "too much", "lot of work", "many tasks", "pending work"
    ]

    if any(phrase in text for phrase in stress_phrases):
        return [
            {"emotion": "nervousness", "confidence": 0.85},
            {"emotion": "fear", "confidence": 0.65},
            {"emotion": "neutral", "confidence": 0.45}
        ]

    if any(word in text for word in stress_words):
        return [
            {"emotion": "nervousness", "confidence": 0.90},
            {"emotion": "fear", "confidence": 0.80},
            {"emotion": "neutral", "confidence": 0.40}
        ]

    return None

def predict_emotions(text, top_k=3):
    rule_result = rule_based_emotion(text)

    if rule_result:
        return rule_result[:top_k]

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