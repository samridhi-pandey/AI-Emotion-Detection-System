import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from deepface import DeepFace

from audio_emotion.audio_predict import predict_audio_mood
from integration.ai_service import detect_text_emotion


def run_text_detection():
    text = input("\nEnter your text: ")

    result = detect_text_emotion(text)

    print("\nText Emotion Detection Result")
    print("-" * 50)
    print("Input Text:", result["input_text"])

    print("\nTop Predicted Emotions:")
    for pred in result["predictions"]:
        print(f"{pred['emotion']} -> {pred['confidence']}")

    print("\nFinal Detected Emotion:", result["final_emotion"])
    print("-" * 50)


def run_webcam_detection():
    from collections import Counter
    import time

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("\nWebcam started.")
    print("Detecting emotion for 5 seconds...")

    detected_emotions = []
    start_time = time.time()

    while time.time() - start_time < 5:
        ret, frame = cap.read()

        if not ret:
            print("Camera not working.")
            break

        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )

            if isinstance(result, list):
                emotion = result[0]["dominant_emotion"]
            else:
                emotion = result["dominant_emotion"]

            detected_emotions.append(emotion)

            cv2.putText(
                frame,
                f"Detecting: {emotion}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except Exception as e:
            pass

        cv2.imshow("Webcam Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if detected_emotions:
        final_emotion = Counter(detected_emotions).most_common(1)[0][0]

        print("\nWebcam Emotion Detection Result")
        print("-" * 50)
        print("Final Detected Emotion:", final_emotion)
        print("-" * 50)
    else:
        print("No emotion detected.")


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

    print("\nAudio Emotion Detection Result")
    print("-" * 50)
    print("Detected Emotion:", result["detected_emotion"])

    if "confidence" in result:
        print("Confidence:", result["confidence"])

    print("-" * 50)

    os.remove(audio_path)


def main():
    while True:
        print("\nAI Emotion Detection System")
        print("=" * 60)
        print("1. Text Emotion Detection")
        print("2. Webcam Emotion Detection")
        print("3. Live Audio Emotion Detection")
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