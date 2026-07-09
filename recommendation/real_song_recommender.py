import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "backend" / "data.csv"

FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness"
]

MOOD_PROFILES = {
    "Happy": [0.8, 0.75, 0.8, 120, 0.1],
    "Sad": [0.35, 0.3, 0.25, 75, 0.6],
    "Calm": [0.45, 0.35, 0.5, 85, 0.7],
    "Energetic": [0.75, 0.9, 0.7, 140, 0.05],
    "Motivated": [0.7, 0.85, 0.65, 130, 0.1],
    "Romantic": [0.55, 0.45, 0.6, 95, 0.5]
}


def _load_and_prepare_dataset():
    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURES + [
        "song_name",
        "artist_name",
        "spotify_track_link",
        "thumbnail_link"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=required_columns).reset_index(drop=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURES].values)

    return df, scaler, scaled_features


# Loaded ONCE at server startup, reused for every request
_DF, _SCALER, _SCALED_FEATURES = _load_and_prepare_dataset()


def recommend_songs(mood, top_n=5):
    mood = mood.strip().capitalize()

    if mood not in MOOD_PROFILES:
        raise ValueError(
            f"Invalid mood '{mood}'. Choose from {list(MOOD_PROFILES.keys())}"
        )

    mood_vector = np.array(MOOD_PROFILES[mood]).reshape(1, -1)
    scaled_mood_vector = _SCALER.transform(mood_vector)

    similarity_scores = cosine_similarity(scaled_mood_vector, _SCALED_FEATURES)[0]

    df = _DF.copy()
    df["similarity_score"] = similarity_scores

    recommended_df = df.sort_values(by="similarity_score", ascending=False).head(top_n)

    recommendations = []
    for _, row in recommended_df.iterrows():
        recommendations.append({
            "mood": mood,
            "track_name": row["song_name"],
            "artists": row["artist_name"],
            "spotify_track_link": row["spotify_track_link"],
            "thumbnail_link": row["thumbnail_link"],
            "similarity_score": round(float(row["similarity_score"]), 4),
            "danceability": round(float(row["danceability"]), 3),
            "energy": round(float(row["energy"]), 3),
            "valence": round(float(row["valence"]), 3),
            "tempo": round(float(row["tempo"]), 2),
            "acousticness": round(float(row["acousticness"]), 3)
        })

    return recommendations


if __name__ == "__main__":
    mood = input("Enter mood: ").strip()
    songs = recommend_songs(mood)
    for s in songs:
        print(s)