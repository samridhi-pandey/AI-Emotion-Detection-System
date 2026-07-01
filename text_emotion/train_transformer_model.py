import numpy as np
import pandas as pd
from pathlib import Path

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "datasets" / "text_emotion" / "goemotions"
TRAIN_PATH = DATA_DIR / "train.tsv"
DEV_PATH = DATA_DIR / "dev.tsv"
TEST_PATH = DATA_DIR / "test.tsv"
EMOTIONS_PATH = DATA_DIR / "emotions.txt"

MODEL_DIR = Path(__file__).resolve().parent / "transformer_model"

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128

TRAIN_SAMPLE_SIZE = 8000
DEV_SAMPLE_SIZE = 1500
TEST_SAMPLE_SIZE = 1500


def load_emotions():
    with open(EMOTIONS_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


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

    return df[["text", "label_list"]]


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probabilities >= 0.5).astype(int)

    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    h_loss = hamming_loss(labels, predictions)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": h_loss
    }


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)

    emotions = load_emotions()

    train_df = load_dataset(TRAIN_PATH)
    dev_df = load_dataset(DEV_PATH)
    test_df = load_dataset(TEST_PATH)



    mlb = MultiLabelBinarizer(classes=list(range(len(emotions))))

    train_labels = mlb.fit_transform(train_df["label_list"]).astype("float32")
    dev_labels = mlb.transform(dev_df["label_list"]).astype("float32")
    test_labels = mlb.transform(test_df["label_list"]).astype("float32")

    train_df["labels"] = train_labels.tolist()
    dev_df["labels"] = dev_labels.tolist()
    test_df["labels"] = test_labels.tolist()

    train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
    dev_dataset = Dataset.from_pandas(dev_df[["text", "labels"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "labels"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_data(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_dataset = train_dataset.map(tokenize_data, batched=True)
    dev_dataset = dev_dataset.map(tokenize_data, batched=True)
    test_dataset = test_dataset.map(tokenize_data, batched=True)

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=False
    )

    dev_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=False
    )

    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(emotions),
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        warmup_ratio=0.1,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    print("Training transformer model...")
    trainer.train()

    print("\nTesting transformer model...")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    print("\nSaving transformer model...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    with open(MODEL_DIR / "emotions.txt", "w", encoding="utf-8") as file:
        for emotion in emotions:
            file.write(emotion + "\n")

    print("\nTransformer model saved at:", MODEL_DIR)