import pandas as pd

DATASET_PATH = "datasets/music/278k_song_labelled.csv"

df = pd.read_csv(DATASET_PATH)

print(df.columns.tolist())
print(df.head())