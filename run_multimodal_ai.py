import os
import sys
import cv2
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_emotion.text_predict import predict_emotions
from facial_emotion.face_predict import predict_face_emotion
from audio_emotion.audio_predict import predict_audio_mood


# ----------------------------
# TEXT EMOTION DETECTION
# ----------------------------
def run_text_detection():

    from integration.emotion_mapper import get_final_text_emotion

    text = input("\nEnter your text: ")

    predictions = predict_emotions(text)

    print("\n========== TEXT EMOTION DETECTION ==========\n")
    print("Top Predicted Emotions:\n")

    for i, pred in enumerate(predictions, start=1):
        print(
            f"{i}. {pred['emotion']} "
            f"(Confidence: {pred['confidence']:.4f})"
        )

    final_emotion = get_final_text_emotion(predictions)

    print("\nFinal Emotion:", final_emotion)


# ----------------------------
# WEBCAM EMOTION DETECTION
# ----------------------------
def run_webcam_detection():

    cap = cv2.VideoCapture(0)

    print("\nWebcam started.")
    print("Press Q to quit.\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to access webcam.")
            break

        # Predict emotion
        result = predict_face_emotion(frame)

        if result is not None:
            from integration.emotion_mapper import FACE_TO_FINAL

            x, y, w, h = result["box"]

            final_emotion = FACE_TO_FINAL.get(
                result["emotion"],
                "Neutral"
            )

            confidence = result["confidence"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{final_emotion} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Facial Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# AUDIO RECORDING
# ----------------------------
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

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    write(
        temp.name,
        sample_rate,
        (audio * 32767).astype("int16")
    )

    return temp.name


# ----------------------------
# AUDIO EMOTION DETECTION
# ----------------------------
def run_audio_detection():

    audio_path = record_audio()

    from integration.emotion_mapper import AUDIO_TO_FINAL

    result = predict_audio_mood(audio_path)

    raw_emotion = result["detected_emotion"]

    final_emotion = AUDIO_TO_FINAL.get(
        raw_emotion,
        "Neutral"
    )

    print("\n========== AUDIO EMOTION DETECTION ==========\n")
    print("Detected Emotion:", final_emotion)

    os.remove(audio_path)


# ----------------------------
# MAIN MENU
# ----------------------------
def main():

    while True:

        print("\n")
        print("=" * 55)
        print(" AI MULTI-MODAL EMOTION DETECTION SYSTEM ")
        print("=" * 55)

        print("1. Text Emotion Detection")
        print("2. Facial Emotion Detection (Webcam)")
        print("3. Audio Emotion Detection (Microphone)")
        print("4. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            run_text_detection()

        elif choice == "2":
            run_webcam_detection()

        elif choice == "3":
            run_audio_detection()

        elif choice == "4":
            print("\nExiting...")
            break

        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()