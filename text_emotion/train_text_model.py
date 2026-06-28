import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, hamming_loss, classification_report
from sklearn.model_selection import GridSearchCV


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

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                solver="liblinear"
            )
        ))
    ])

    param_grid = {
        "tfidf__max_features": [20000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__estimator__C": [0.1, 1, 10],
        "classifier__estimator__class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_micro",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    print("Starting hyperparameter tuning...")
    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest Cross Validation Score:")
    print(grid.best_score_)

    model = grid.best_estimator_

    print("\nTesting best model...")
    y_pred = model.predict(X_test)

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    h_loss = hamming_loss(y_test, y_pred)

    print("\nFinal Model Evaluation")
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

    print("\nTuned model saved at:", MODEL_PATH)
    print("Label binarizer saved at:", BINARIZER_PATH)


if __name__ == "__main__":
    train_model()