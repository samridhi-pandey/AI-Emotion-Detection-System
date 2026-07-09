import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from deepface import DeepFace

from recommendation.real_song_recommender import recommend_songs
from audio_emotion.audio_predict import predict_audio_mood
from integration.ai_service import recommend_music_from_text


FACE_TO_MOOD = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Energetic",
    "fear": "Calm",
    "surprise": "Energetic",
    "neutral": "Calm",
    "disgust": "Calm"
}


def display_songs(songs):
    print("\nRecommended Songs:\n")

    for i, song in enumerate(songs, start=1):
        print(f"{i}. {song['track_name']}")
        print(f"   Artist: {song['artists']}")
        print(f"   Genre: {song['genre']}")
        print(f"   Popularity: {song['popularity']}")
        print(f"   Similarity Score: {song['similarity_score']}")
        print("-" * 50)


def run_text_detection():
    text = input("\nEnter your text: ")

    result = recommend_music_from_text(text)

    print("\nText Emotion Recommendation Result")
    print("-" * 50)
    print("Input Text:", result["input_text"])
    print("Mapped Mood:", result["mapped_mood"])

    display_songs(result["recommendations"])


def run_webcam_detection():
    cap = cv2.VideoCapture(0)

    print("\nWebcam started.")
    print("Press R to recommend songs")
    print("Press Q to quit webcam")

    last_emotion = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera not working.")
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            emotion = result[0]["dominant_emotion"]
            last_emotion = emotion
            mood = FACE_TO_MOOD.get(emotion, "Calm")

            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Mood: {mood}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

        except Exception as e:
            print("Face detection error:", e)

        cv2.imshow("Webcam Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if last_emotion is None:
                print("No emotion detected yet.")
            else:
                mood = FACE_TO_MOOD.get(last_emotion, "Calm")

                print("\nWebcam Emotion Recommendation Result")
                print("-" * 50)
                print("Detected Emotion:", last_emotion)

                songs = recommend_songs(mood, top_n=5)
                display_songs(songs)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


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


def run_audio_detection():
    audio_path = record_audio(duration=5)

    result = predict_audio_mood(audio_path)
    emotion = result["detected_emotion"]

    print("\nAudio Emotion Recommendation Result")
    print("-" * 50)
    print("Detected Emotion:", emotion)

    if emotion == "happy":
        mood = "Happy"

    elif emotion == "sad":
        mood = "Sad"

    elif emotion in ["angry", "surprised"]:
        mood = "Energetic"

    else:
        mood = "Calm"

    songs = recommend_songs(mood, top_n=5)
    display_songs(songs)

    os.remove(audio_path)


def main():
    while True:
        print("\nAI Emotion-Based Music Recommendation System")
        print("=" * 60)
        print("1. Text Emotion Detection + Song Recommendation")
        print("2. Webcam Emotion Detection + Song Recommendation")
        print("3. Live Audio Emotion Detection + Song Recommendation")
        print("4. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            run_text_detection()

        elif choice == "2":
            run_webcam_detection()

        elif choice == "3":
            run_audio_detection()

        elif choice == "4":
            print("Exiting system.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()