from fastapi import FastAPI
from pydantic import BaseModel
from integration.ai_service import recommend_music_from_text

app = FastAPI(
    title="AI Emotion-Based Music Recommendation System",
    description="Text emotion to mood-based real song recommendation system",
    version="1.0"
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "AI Emotion-Based Music Recommendation System is running"
    }


@app.post("/recommend/text")
def recommend_text(request: TextRequest):
    result = recommend_music_from_text(request.text)
    return result