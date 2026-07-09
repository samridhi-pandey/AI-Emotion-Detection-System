import sys
sys.path.append(".")

from text_emotion.text_predict import predict_emotions
from recommendation.mood_mapper import map_emotions_to_mood


def detect_text_emotion(text):
    predictions = predict_emotions(text, top_k=3)
    mood_result = map_emotions_to_mood(predictions)

    return {
        "input_text": text,
        "predictions": predictions,
        "final_emotion": mood_result["final_mood"],
        "mood_scores": mood_result["mood_scores"]
    }