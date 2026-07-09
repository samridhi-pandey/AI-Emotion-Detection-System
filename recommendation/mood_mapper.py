GOEMOTION_TO_MOOD = {
    "admiration": "Motivated",
    "amusement": "Happy",
    "anger": "Energetic",
    "annoyance": "Energetic",
    "approval": "Motivated",
    "caring": "Calm",
    "confusion": "Calm",
    "curiosity": "Motivated",
    "desire": "Romantic",
    "disappointment": "Sad",
    "disapproval": "Energetic",
    "disgust": "Energetic",
    "embarrassment": "Calm",
    "excitement": "Happy",
    "fear": "Calm",
    "gratitude": "Happy",
    "grief": "Sad",
    "joy": "Happy",
    "love": "Romantic",
    "nervousness": "Calm",
    "optimism": "Motivated",
    "pride": "Motivated",
    "realization": "Calm",
    "relief": "Calm",
    "remorse": "Sad",
    "sadness": "Sad",
    "surprise": "Motivated",
    "neutral": "Calm"
}


def map_emotions_to_mood(predicted_emotions):
    mood_scores = {}

    for item in predicted_emotions:
        emotion = item["emotion"]
        confidence = item["confidence"]

        mood = GOEMOTION_TO_MOOD.get(emotion, "Neutral")

        if mood not in mood_scores:
            mood_scores[mood] = 0

        mood_scores[mood] += confidence

    final_mood = max(mood_scores, key=mood_scores.get)

    return {
        "final_mood": final_mood,
        "mood_scores": mood_scores
    }