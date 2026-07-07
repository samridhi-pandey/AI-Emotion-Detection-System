from fastapi import FastAPI
from recommendation import recommend_songs
from database import history_collection
from datetime import datetime

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Emotion Music Recommendation Backend Running"
    }


@app.post("/recommend")
def recommend(data: dict):

    emotion = data.get("emotion", "neutral")

    songs = recommend_songs(emotion)

    history_collection.insert_one({
        "emotion": emotion,
        "songs": songs,
        "timestamp": datetime.now().isoformat()
    })

    return {
        "emotion": emotion,
        "recommended_songs": songs
    }
@app.get("/history")
def get_history():


    return list(
    history_collection.find({}, {"_id": 0})
    .sort("timestamp", -1)
)
from collections import Counter


@app.get("/stats")
def get_stats():

    emotions = []

    for item in history_collection.find():
        emotions.append(item["emotion"])

    return dict(Counter(emotions))
