import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from facial_emotion.face_predict import predict_face_emotion
from recommendation.real_song_recommender import recommend_songs


EMOTION_TO_MOOD = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Energetic",
    "surprise": "Energetic",
    "fear": "Calm",
    "neutral": "Calm",
    "disgust": "Calm"
}


cap = cv2.VideoCapture(0)

print("Press R to recommend songs")
print("Press Q to quit")

last_emotion = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = predict_face_emotion(frame)

    if result is not None:

        emotion = result["emotion"]
        confidence = result["confidence"]

        last_emotion = emotion

        x, y, w, h = result["box"]

        mood = EMOTION_TO_MOOD.get(emotion, "Calm")

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Mood: {mood}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "FER2013 Facial Emotion Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

        if last_emotion is None:
            print("No face detected.")
        else:
            mood = EMOTION_TO_MOOD.get(last_emotion, "Calm")

            print("\nDetected Emotion:", last_emotion)

            songs = recommend_songs(mood, top_n=5)

            print("\nRecommended Songs:\n")

            for i, song in enumerate(songs, start=1):
                print(f"{i}. {song['track_name']}")
                print(f"   Artist: {song['artists']}")
                print(f"   Genre: {song['genre']}")
                print("-" * 50)

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()