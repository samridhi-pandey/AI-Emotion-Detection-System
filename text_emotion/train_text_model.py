import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, hamming_loss, classification_report


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "datasets" / "text_emotion" / "goemotions"
TRAIN_PATH = DATA_DIR / "train.tsv"
DEV_PATH = DATA_DIR / "dev.tsv"
TEST_PATH = DATA_DIR / "test.tsv"
EMOTIONS_PATH = DATA_DIR / "emotions.txt"

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "goemotions_text_model.pkl"
BINARIZER_PATH = MODEL_DIR / "label_binarizer.pkl"


def load_emotions():
    with open(EMOTIONS_PATH, "r", encoding="utf-8") as file:
        emotions = [line.strip() for line in file.readlines()]
    return emotions


def load_dataset(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "labels", "id"]
    )

    df["label_list"] = df["labels"].apply(
        lambda x: [int(label) for label in str(x).split(",")]
    )

    return df


def train_model():
    MODEL_DIR.mkdir(exist_ok=True)

    emotions = load_emotions()

    train_df = load_dataset(TRAIN_PATH)
    dev_df = load_dataset(DEV_PATH)
    test_df = load_dataset(TEST_PATH)

    train_df = pd.concat([train_df, dev_df], ignore_index=True)

    X_train = train_df["text"]
    X_test = test_df["text"]

    y_train_labels = train_df["label_list"]
    y_test_labels = test_df["label_list"]

    mlb = MultiLabelBinarizer(classes=list(range(len(emotions))))

    y_train = mlb.fit_transform(y_train_labels)
    y_test = mlb.transform(y_test_labels)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    print("Testing model...")
    y_pred = model.predict(X_test)

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    h_loss = hamming_loss(y_test, y_pred)

    print("\nModel Evaluation")
    print("Micro F1 Score:", micro_f1)
    print("Macro F1 Score:", macro_f1)
    print("Hamming Loss:", h_loss)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=emotions,
        zero_division=0
    ))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(mlb, BINARIZER_PATH)

    print("\nModel saved at:", MODEL_PATH)
    print("Label binarizer saved at:", BINARIZER_PATH)


if __name__ == "__main__":
    train_model()