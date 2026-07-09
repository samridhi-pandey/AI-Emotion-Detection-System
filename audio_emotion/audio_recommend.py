import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_emotion.audio_predict import predict_audio_mood
from recommendation.real_song_recommender import recommend_songs


def display_audio_recommendations(audio_path):
    result = predict_audio_mood(audio_path)

    emotion = result["detected_emotion"]
    mood = result["mapped_mood"]

    print("\nAudio Emotion Detection Result")
    print("-" * 50)
    print("Detected Emotion:", emotion)
    print("Mapped Mood:", mood)

    print("\nExtracted Audio Features:")
    for key, value in result["features"].items():
        print(f"{key}: {round(value, 4)}")

    songs = recommend_songs(mood, top_n=5)

    print("\nRecommended Songs:\n")

    for i, song in enumerate(songs, start=1):
        print(f"{i}. {song['track_name']}")
        print(f"   Artist: {song['artists']}")
        print(f"   Genre: {song['genre']}")
        print(f"   Popularity: {song['popularity']}")
        print(f"   Similarity Score: {song['similarity_score']}")
        print("-" * 50)


if __name__ == "__main__":
    audio_path = input("Enter audio file path: ").strip()
    display_audio_recommendations(audio_path)