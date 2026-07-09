from collections import Counter

TEXT_TO_FINAL = {
    # Happy
    "joy": "Happy",
    "excitement": "Happy",
    "optimism": "Happy",
    "admiration": "Happy",
    "approval": "Happy",
    "gratitude": "Happy",
    "love": "Happy",
    "caring": "Happy",
    "amusement": "Happy",
    "pride": "Happy",
    "desire": "Happy",
    "relief": "Happy",

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

    # Fear
    "fear": "Fear",
    "nervousness": "Fear",

    # Surprise
    "surprise": "Surprise",
    "realization": "Surprise",
    "confusion": "Surprise",
    "curiosity": "Surprise",

    # Disgust
    "disgust": "Disgust",

    # Neutral
    "neutral": "Neutral"
}


def get_final_text_emotion(predictions):
    """
    Convert top predicted emotions into one final emotion.
    """
    mapped = []

    for pred in predictions:
        emotion = pred["emotion"]
        mapped.append(TEXT_TO_FINAL.get(emotion, "Neutral"))

    counter = Counter(mapped)
    return counter.most_common(1)[0][0]


FACE_TO_FINAL = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fear",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "neutral": "Neutral"
}

AUDIO_TO_FINAL = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fearful": "Fear",
    "fear": "Fear",
    "surprised": "Surprise",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "calm": "Neutral",
    "neutral": "Neutral"
}