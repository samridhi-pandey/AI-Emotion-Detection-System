import sys
import os
import tempfile
from integration.emotion_mapper import AUDIO_TO_FINAL

final = AUDIO_TO_FINAL[result["detected_emotion"]]
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
from scipy.io.wavfile import write

from audio_emotion.audio_predict import predict_audio_mood
from recommendation.real_song_recommender import recommend_songs


def record_audio(duration=5, sample_rate=22050):
    print("\nRecording started...")
    print("Speak now...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("Recording completed.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    write(
        temp_file.name,
        sample_rate,
        (audio * 32767).astype("int16")
    )

    return temp_file.name


def recommend_from_live_audio():
    audio_path = record_audio(duration=5)

    result = predict_audio_mood(audio_path)

    print("\nAudio Emotion Detection Result")
    print("-" * 50)
    print("Detected Emotion:", result["detected_emotion"])
    emotion = result["detected_emotion"]

    print("Detected Emotion:", emotion)

    if emotion in ["happy"]:
        mood = "Happy"

    elif emotion in ["sad"]:
        mood = "Sad"

    elif emotion in ["angry", "surprised"]:
        mood = "Energetic"

    else:
        mood = "Calm"

    songs = recommend_songs(mood, top_n=5)

    print("\nRecommended Songs:\n")

    for i, song in enumerate(songs, start=1):
        print(f"{i}. {song['track_name']}")
        print(f"   Artist: {song['artists']}")
        print(f"   Genre: {song['genre']}")
        print(f"   Popularity: {song['popularity']}")
        print(f"   Similarity Score: {song['similarity_score']}")
        print("-" * 50)

    os.remove(audio_path)


if __name__ == "__main__":
    recommend_from_live_audio()