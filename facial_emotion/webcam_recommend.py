import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
from deepface import DeepFace
from recommendation.real_song_recommender import recommend_songs


FACE_TO_MOOD = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Energetic",
    "fear": "Calm",
    "surprise": "Energetic",
    "neutral": "Calm",
    "disgust": "Calm"
}


cap = cv2.VideoCapture(0)

current_emotion = "Press S to scan"
current_mood = ""

print("Press S to scan emotion and recommend songs")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    cv2.putText(
        frame,
        f"Emotion: {current_emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if current_mood:
        cv2.putText(
            frame,
            f"Mood: {current_mood}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("Webcam Emotion Music Recommendation", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("\nScanning emotion...")

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            current_emotion = result[0]["dominant_emotion"]
            current_mood = FACE_TO_MOOD.get(current_emotion, "Calm")

            print("Detected Emotion:", current_emotion)
            print("Mapped Mood:", current_mood)

            songs = recommend_songs(current_mood, top_n=5)

            print("\nRecommended Songs:\n")

            for i, song in enumerate(songs, start=1):
                print(f"{i}. {song['track_name']}")
                print(f"   Artist: {song['artists']}")
                print(f"   Genre: {song['genre']}")
                print(f"   Popularity: {song['popularity']}")
                print(f"   Similarity Score: {song['similarity_score']}")
                print("-" * 50)

        except Exception as e:
            current_emotion = "Detection failed"
            current_mood = ""
            print("Error:", e)

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()