import pandas as pd

songs_df = pd.read_csv("data.csv")


def recommend_songs(emotion):

    if emotion == "happy":
        filtered = songs_df[
            (songs_df["valence"] > 0.7) &
            (songs_df["energy"] > 0.6)
        ]

    elif emotion == "sad":
        filtered = songs_df[
            (songs_df["valence"] < 0.4) &
            (songs_df["energy"] < 0.5)
        ]

    elif emotion == "angry":
        filtered = songs_df[
            songs_df["energy"] > 0.8
        ]

    elif emotion == "relaxed":
        filtered = songs_df[
            songs_df["energy"] < 0.5
        ]

    else:
        filtered = songs_df

    recommendations = filtered.sample(
        n=min(5, len(filtered))
    )[["song_name", "artist_name"]]

    return recommendations.to_dict(orient="records")
# import pandas as pd

# # Load dataset
# songs_df = pd.read_csv("spotify_synthetic.csv")


# def recommend_songs(emotion):

#     if emotion == "happy":
#         filtered = songs_df[
#             (songs_df['valence'] > 0.7) &
#             (songs_df['energy'] > 0.6)
#         ]

#     elif emotion == "sad":
#         filtered = songs_df[
#             (songs_df['valence'] < 0.4) &
#             (songs_df['energy'] < 0.5)
#         ]

#     elif emotion == "angry":
#         filtered = songs_df[
#             (songs_df['energy'] > 0.8)
#         ]

#     elif emotion == "relaxed":
#         filtered = songs_df[
#             (songs_df['energy'] < 0.5)
#         ]

#     else:
#         filtered = songs_df

#     recommendations = filtered[
#         ['track_name', 'artists']
#     ].head(5)

#     return recommendations.to_dict(orient='records')