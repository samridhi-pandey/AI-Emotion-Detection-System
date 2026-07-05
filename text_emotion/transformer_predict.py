"""
train_transformer_model.py

RoBERTa Fine-Tuning for GoEmotions
(Custom PyTorch Training Loop)

Author: Samridhi Pandey
"""

from pathlib import Path
import shutil
import random
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    hamming_loss,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "roberta-base"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

PATIENCE = 2

THRESHOLD = 0.30

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR /
    "datasets" /
    "text_emotion" /
    "goemotions"
)

MODEL_DIR = (
    Path(__file__).resolve().parent /
    "transformer_model"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed()

# ==========================================================
# Load Dataset
# ==========================================================

def load_emotions():

    with open(
        DATA_DIR / "emotions.txt",
        encoding="utf-8",
    ) as f:

        emotions = [
            line.strip()
            for line in f
        ]

    return emotions


def load_split(split):

    df = pd.read_csv(
        DATA_DIR / f"{split}.tsv",
        sep="\t",
        header=None,
        names=[
            "text",
            "labels",
            "id",
        ],
    )

    df["label_list"] = df["labels"].apply(
        lambda x: [
            int(i)
            for i in str(x).split(",")
        ]
    )

    return df

# ==========================================================
# Dataset Class
# ==========================================================

class GoEmotionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        labels,
    ):

        self.texts = dataframe["text"].tolist()

        self.labels = labels

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(

            self.texts[idx],

            truncation=True,

            padding="max_length",

            max_length=MAX_LENGTH,

            return_tensors="pt",

        )

        return {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.float32,
                ),

        }

    # ==========================================================
    # Metrics
    # ==========================================================

    def calculate_metrics(logits, labels):
        probabilities = torch.sigmoid(
            torch.tensor(logits)
        ).numpy()

        predictions = (
                probabilities >= THRESHOLD
        ).astype(int)

        return {

            "micro_f1": f1_score(
                labels,
                predictions,
                average="micro",
                zero_division=0,
            ),

            "macro_f1": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),

            "hamming_loss": hamming_loss(
                labels,
                predictions,
            ),

        }

    # ==========================================================
    # Data Preparation
    # ==========================================================

    print("Loading GoEmotions...")

    emotions = load_emotions()

    train_df = load_split("train")

    dev_df = load_split("dev")

    test_df = load_split("test")

    mlb = MultiLabelBinarizer(
        classes=list(range(len(emotions)))
    )

    mlb.fit(train_df["label_list"])

    train_labels = mlb.transform(
        train_df["label_list"]
    ).astype(np.float32)

    dev_labels = mlb.transform(
        dev_df["label_list"]
    ).astype(np.float32)

    test_labels = mlb.transform(
        test_df["label_list"]
    ).astype(np.float32)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = GoEmotionDataset(
        train_df,
        tokenizer,
        train_labels,
    )

    dev_dataset = GoEmotionDataset(
        dev_df,
        tokenizer,
        dev_labels,
    )

    test_dataset = GoEmotionDataset(
        test_df,
        tokenizer,
        test_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Train Samples      : {len(train_dataset)}")
    print(f"Validation Samples : {len(dev_dataset)}")
    print(f"Test Samples       : {len(test_dataset)}")

    # ==========================================================
    # Model
    # ==========================================================

    print("\nLoading RoBERTa...")

    model = AutoModelForSequenceClassification.from_pretrained(

        MODEL_NAME,

        num_labels=len(emotions),

        problem_type="multi_label_classification",

    )

    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    total_steps = len(train_loader) * EPOCHS

    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=warmup_steps,

        num_training_steps=total_steps,

    )

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=torch.cuda.is_available(),

    )

    # ==========================================================
    # Training
    # ==========================================================

    def train_one_epoch():
        model.train()

        running_loss = 0.0

        progress = tqdm(

            train_loader,

            desc="Training",

            leave=False,

        )

        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(

                    "cuda",

                    enabled=torch.cuda.is_available(),

            ):
                outputs = model(

                    input_ids=input_ids,

                    attention_mask=attention_mask,

                )

                loss = criterion(

                    outputs.logits,

                    labels,

                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            running_loss += loss.item()

            progress.set_postfix(

                loss=f"{loss.item():.4f}"

            )

        return running_loss / len(train_loader)

    # ==========================================================
    # Validation
    # ==========================================================

    @torch.no_grad()
    def evaluate(data_loader):

        model.eval()

        running_loss = 0.0

        all_logits = []

        all_labels = []

        progress = tqdm(
            data_loader,
            desc="Evaluating",
            leave=False,
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available(),
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits

                loss = criterion(
                    logits,
                    labels,
                )

            running_loss += loss.item()

            all_logits.append(
                logits.detach().cpu()
            )

            all_labels.append(
                labels.detach().cpu()
            )

        all_logits = torch.cat(all_logits).numpy()

        all_labels = torch.cat(all_labels).numpy()

        metrics = calculate_metrics(
            all_logits,
            all_labels,
        )

        return (
            running_loss / len(data_loader),
            metrics,
        )

    # ==========================================================
    # Model Saving
    # ==========================================================

    def save_model():

        model.save_pretrained(
            MODEL_DIR
        )

        tokenizer.save_pretrained(
            MODEL_DIR
        )

        shutil.copy(
            DATA_DIR / "emotions.txt",
            MODEL_DIR / "emotions.txt",
        )

    # ==========================================================
    # Main Training Loop
    # ==========================================================

    def main():

        print("=" * 70)
        print("RoBERTa Fine-Tuning")
        print("=" * 70)

        best_micro = -1.0

        patience_counter = 0

        for epoch in range(EPOCHS):

            print()

            print(f"Epoch {epoch + 1}/{EPOCHS}")

            train_loss = train_one_epoch()

            val_loss, metrics = evaluate(
                dev_loader
            )

            print(f"Train Loss     : {train_loss:.4f}")

            print(f"Valid Loss     : {val_loss:.4f}")

            print(
                f"Micro F1       : "
                f"{metrics['micro_f1']:.4f}"
            )

            print(
                f"Macro F1       : "
                f"{metrics['macro_f1']:.4f}"
            )

            print(
                f"Hamming Loss   : "
                f"{metrics['hamming_loss']:.4f}"
            )

            if metrics["micro_f1"] > best_micro:

                best_micro = metrics["micro_f1"]

                patience_counter = 0

                save_model()

                print()

                print("Best model saved.")

            else:

                patience_counter += 1

                print()

                print(
                    f"No improvement "
                    f"({patience_counter}/{PATIENCE})"
                )

                if patience_counter >= PATIENCE:
                    print()

                    print("Early stopping.")

                    break

        print()

        print("=" * 70)

        print("Loading Best Model")

        print("=" * 70)

        best_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        best_model.to(DEVICE)

        model.load_state_dict(
            best_model.state_dict()
        )

        test_loss, metrics = evaluate(
            test_loader
        )

        print()

        print("=" * 70)

        print("Final Test Results")

        print("=" * 70)

        print(f"Test Loss      : {test_loss:.4f}")

        print(
            f"Micro F1       : "
            f"{metrics['micro_f1']:.4f}"
        )

        print(
            f"Macro F1       : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Hamming Loss   : "
            f"{metrics['hamming_loss']:.4f}"
        )

        print()

        print("Model saved at:")

        print(MODEL_DIR)

    if __name__ == "__main__":
        main()

        """
        transformer_predict.py

        Inference for RoBERTa GoEmotions Model

        Author: Samridhi Pandey
        """

        from pathlib import Path

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        # ==========================================================
        # Configuration
        # ==========================================================

        BASE_DIR = Path(__file__).resolve().parent

        MODEL_DIR = BASE_DIR / "transformer_model"

        THRESHOLD = 0.30

        DEVICE = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ==========================================================
        # Load Emotion Labels
        # ==========================================================

        with open(
                MODEL_DIR / "emotions.txt",
                encoding="utf-8",
        ) as f:

            EMOTIONS = [
                line.strip()
                for line in f
            ]

        # ==========================================================
        # Load Tokenizer
        # ==========================================================

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        # ==========================================================
        # Load Model
        # ==========================================================

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        model.to(DEVICE)

        model.eval()

        # ==========================================================
        # Prediction Function
        # ==========================================================

        @torch.no_grad()
        def predict(text):

            encoding = tokenizer(

                text,

                truncation=True,

                padding=True,

                max_length=128,

                return_tensors="pt",

            )

            encoding = {

                k: v.to(DEVICE)

                for k, v in encoding.items()

            }

            outputs = model(**encoding)

            probabilities = torch.sigmoid(
                outputs.logits
            ).cpu().numpy()[0]

            predictions = []

            for emotion, score in zip(

                    EMOTIONS,

                    probabilities,

            ):

                if score >= THRESHOLD:
                    predictions.append(

                        (

                            emotion,

                            float(score),

                        )

                    )

            predictions.sort(

                key=lambda x: x[1],

                reverse=True,

            )

            return predictions

        # ==========================================================
        # Example
        # ==========================================================

        if __name__ == "__main__":

            while True:

                text = input("\nEnter text: ")

                if text.lower() == "exit":
                    break

                result = predict(text)

                if not result:
                    print("\nNo emotion detected.\n")

                    continue

                print()

                for emotion, score in result:
                    print(

                        f"{emotion:<20}"

                        f"{score:.4f}"

                    )