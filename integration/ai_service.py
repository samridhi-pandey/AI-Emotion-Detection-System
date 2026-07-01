from text_emotion.transformer_predict import predict_emotions


def detect_text_emotion(text):

    result = predict_emotions(text)

    return {

        "input_text": text,

        "predictions": result["predictions"],

        "final_emotion": result["final_emotion"],

        "final_confidence": result["final_confidence"]

    }
def detect_text_emotion(text):
    result = predict_emotions(text)

    return {
        "input_text": text,
        "predictions": result["predictions"],
        "final_emotion": result["final_emotion"]
    }

EMOTION_MAPPING = {

    # ---------------- Happy ----------------
    "joy": "Happy",
    "amusement": "Happy",
    "gratitude": "Happy",
    "love": "Happy",
    "optimism": "Happy",
    "pride": "Happy",
    "relief": "Happy",
    "admiration": "Happy",
    "approval": "Happy",
    "excitement": "Happy",

    # ---------------- Sad ----------------
    "sadness": "Sad",
    "grief": "Sad",
    "disappointment": "Sad",
    "remorse": "Sad",
    "embarrassment": "Sad",

    # ---------------- Angry ----------------
    "anger": "Angry",
    "annoyance": "Angry",
    "disapproval": "Angry",

    # ---------------- Anxious ----------------
    "fear": "Anxious",
    "nervousness": "Anxious",

    # ---------------- Calm ----------------
    "neutral": "Calm",
    "relief": "Calm",

    # ---------------- Motivated ----------------
    "determination": "Motivated",
    "desire": "Motivated",
    "optimism": "Motivated",

    # ---------------- Interested ----------------
    "curiosity": "Interested",
    "realization": "Interested",

    # ---------------- Caring ----------------
    "caring": "Caring",

    # ---------------- Surprise ----------------
    "surprise": "Surprised",

    # ---------------- Disgust ----------------
    "disgust": "Disgust",

    # ---------------- Confused ----------------
    "confusion": "Confused"
}


def detect_text_emotion(text):
    predictions = predict_emotions(text, top_k=3)

    top_emotion = predictions[0]["emotion"]
    final_emotion = FINAL_EMOTION_MAP.get(top_emotion, top_emotion.title())

    return {
        "input_text": text,
        "predictions": predictions,
        "final_emotion": final_emotion
    }





