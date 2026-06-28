from text_emotion.text_predict import predict_emotions


FINAL_EMOTION_MAP = {
    "joy": "Happy",
    "amusement": "Happy",
    "excitement": "Excited",
    "optimism": "Happy",
    "gratitude": "Happy",
    "love": "Happy",
    "admiration": "Happy",
    "approval": "Happy",
    "caring": "Happy",

    "sadness": "Sad",
    "grief": "Sad",
    "disappointment": "Sad",
    "remorse": "Sad",

    "anger": "Angry",
    "annoyance": "Angry",
    "disapproval": "Angry",
    "disgust": "Angry",

    "fear": "Anxious",
    "nervousness": "Anxious",

    "curiosity": "Excited",
    "surprise": "Surprised",
    "confusion": "Confused",

    "neutral": "Neutral",
    "realization": "Neutral",
    "desire": "Excited",
    "relief": "Happy",
    "embarrassment": "Anxious",
    "pride": "Happy"
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





