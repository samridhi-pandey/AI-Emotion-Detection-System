import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "transformer_model"

EMOTIONS_PATH = (
    BASE_DIR.parent
    / "datasets"
    / "text_emotion"
    / "goemotions"
    / "emotions.txt"
)

# ============================================================
# Load model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

# ============================================================
# Emotion Labels
# ============================================================

with open(EMOTIONS_PATH, "r", encoding="utf-8") as f:
    emotions = [line.strip() for line in f.readlines()]

# ============================================================
# Mapping GoEmotions → Final Emotion
# ============================================================

EMOTION_MAPPING = {

    # Happy
    "joy": "Happy",
    "amusement": "Happy",
    "approval": "Happy",
    "gratitude": "Happy",
    "love": "Happy",
    "optimism": "Happy",
    "pride": "Happy",
    "relief": "Happy",
    "admiration": "Happy",
    "caring": "Happy",
    "desire": "Happy",
    "excitement": "Happy",

    # Sad
    "sadness": "Sad",
    "grief": "Sad",
    "disappointment": "Sad",
    "remorse": "Sad",
    "embarrassment": "Sad",

    # Angry
    "anger": "Angry",
    "annoyance": "Angry",
    "disapproval": "Angry",

    # Anxious
    "fear": "Anxious",
    "nervousness": "Anxious",
    "confusion": "Anxious",
    "realization": "Anxious",

    # Surprise
    "surprise": "Surprised",
    "curiosity": "Surprised",

    # Calm
    "neutral": "Calm",

    # Disgust
    "disgust": "Disgust",

    # Motivated
    "determination": "Motivated"
}


# ============================================================
# Prediction
# ============================================================

def predict_emotions(text, top_k=3):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)[0]

    # --------------------------------------------------------

    predictions = []

    for emotion, prob in zip(emotions, probabilities):

        predictions.append({
            "raw_emotion": emotion,
            "emotion": EMOTION_MAPPING.get(
                emotion,
                emotion.capitalize()
            ),
            "confidence": float(prob)
        })

    # --------------------------------------------------------
    # Aggregate probabilities
    # --------------------------------------------------------

    aggregated_scores = {}

    for pred in predictions:

        category = pred["emotion"]

        aggregated_scores[category] = (
            aggregated_scores.get(category, 0)
            + pred["confidence"]
        )

    final_emotion = max(
        aggregated_scores,
        key=aggregated_scores.get
    )

    # --------------------------------------------------------
    # Top-k raw predictions
    # --------------------------------------------------------

    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True
    )

    top_predictions = predictions[:top_k]

    # Find confidence of final emotion
    total_score = sum(aggregated_scores.values())

    final_confidence = (
                               aggregated_scores[final_emotion] / total_score
                       ) * 100

    return {

        "predictions": [
            {
                "emotion": p["emotion"],
                "raw_emotion": p["raw_emotion"],
                "confidence": round(p["confidence"], 4)
            }
            for p in top_predictions
        ],

        "final_emotion": final_emotion,

        "final_confidence": round(final_confidence, 2)

    }




# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    while True:

        text = input("\nEnter text (q to quit): ")

        if text.lower() == "q":
            break

        result = predict_emotions(text)

        print("\nTop Predictions\n")

        for pred in result["predictions"]:

            print(
                f"{pred['emotion']} "
                f"({pred['raw_emotion']}) "
                f"-> {pred['confidence']}"
            )

        print("\nFinal Emotion:", result["final_emotion"])