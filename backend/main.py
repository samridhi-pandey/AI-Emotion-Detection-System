import sys
sys.path.append("..")

import base64
import numpy as np
import cv2
import tempfile
import os
from audio_emotion.audio_predict import predict_audio_mood

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from integration.ai_service import detect_text_emotion
from recommendation.real_song_recommender import recommend_songs
from facial_emotion.face_predict import predict_face_emotion
from database import history_collection

import av

def convert_webm_to_wav(input_path, output_path):
    input_container = av.open(input_path)
    audio_stream = input_container.streams.audio[0]

    output_container = av.open(output_path, mode="w")
    output_stream = output_container.add_stream("pcm_s16le", rate=22050)

    for frame in input_container.decode(audio_stream):
        frame.pts = None
        for packet in output_stream.encode(frame):
            output_container.mux(packet)

    for packet in output_stream.encode():
        output_container.mux(packet)

    output_container.close()
    input_container.close()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACE_TO_MOOD = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Energetic",
    "fear": "Calm",
    "surprise": "Motivated",
    "disgust": "Sad",
    "neutral": "Calm"
}
AUDIO_TO_MOOD = {
    "neutral": "Calm",
    "calm": "Calm",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Energetic",
    "fearful": "Calm",
    "disgust": "Sad",
    "surprised": "Motivated"
}


@app.get("/")
def home():
    return {"message": "Emotion Music Recommendation Backend Running"}


@app.post("/api/emotion/text")
def emotion_text(data: dict):
    text = data.get("text", "")
    result = detect_text_emotion(text)

    return {
        "success": True,
        "detectedEmotion": result["final_emotion"],
        "predictions": result["predictions"]
    }


@app.post("/api/emotion/face")
def emotion_face(data: dict):
    image_b64 = data.get("image", "")

    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    image_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result = predict_face_emotion(frame)

    if result is None:
        return {"success": False, "error": "No face detected"}

    mood = FACE_TO_MOOD.get(result["emotion"], "Calm")

    return {
        "success": True,
        "detectedEmotion": mood,
        "confidence": result["confidence"]
    }
@app.post("/api/emotion/voice")
async def emotion_voice(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        contents = await file.read()
        temp_webm.write(contents)
        webm_path = temp_webm.name

    wav_path = webm_path.replace(".webm", ".wav")

    try:
        convert_webm_to_wav(webm_path, wav_path)
        result = predict_audio_mood(wav_path)
        raw_emotion = result["detected_emotion"]
        mood = AUDIO_TO_MOOD.get(raw_emotion, "Calm")

        return {
            "success": True,
            "detectedEmotion": mood,
            "rawEmotion": raw_emotion
        }
    except Exception as e:
        print(f"[ERROR] Voice processing failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(webm_path):
            os.remove(webm_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)


@app.get("/api/music/recommend")
def music_recommend(mood: str = "Neutral"):
    songs = recommend_songs(mood)

    try:
        history_collection.insert_one({
            "emotion": mood,
            "songs": songs,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"[WARNING] Could not save to MongoDB: {e}")

    return songs


@app.get("/api/history")
def get_history():
    return list(history_collection.find({}, {"_id": 0}).sort("timestamp", -1))