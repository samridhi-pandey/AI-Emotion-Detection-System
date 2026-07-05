"""
train_transformer_model.py

RoBERTa Fine-Tuning for GoEmotions
Multi-label Emotion Classification
Custom PyTorch Training Loop

Author: Samridhi Pandey
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    hamming_loss,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "datasets"
    / "text_emotion"
    / "goemotions"
)

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "transformer_model"
)

MODEL_NAME = "roberta-base"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

PATIENCE = 2

THRESHOLD = 0.30

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# Utility Functions
# ============================================================


def load_emotions():

    with open(
        DATA_DIR / "emotions.txt",
        encoding="utf-8"
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


# ============================================================
# Dataset
# ============================================================

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


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    logits,
    labels,
):

    probabilities = 1 / (
        1 +
        np.exp(-logits)
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    micro = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    hamming = hamming_loss(
        labels,
        predictions,
    )

    return {

        "micro_f1": micro,

        "macro_f1": macro,

        "hamming_loss": hamming,

    }


# ============================================================
# Prepare Dataset
# ============================================================

print("Loading GoEmotions...")

emotions = load_emotions()

train_df = load_split("train")

dev_df = load_split("dev")

test_df = load_split("test")

mlb = MultiLabelBinarizer(
    classes=list(
        range(len(emotions))
    )
)

mlb.fit(
    train_df["label_list"]
)

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
    num_workers=2,
    pin_memory=True,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"Train Samples : {len(train_dataset)}")
print(f"Validation    : {len(dev_dataset)}")
print(f"Test Samples  : {len(test_dataset)}")
print()

# ============================================================
# Model
# ============================================================

print("Loading RoBERTa model...")

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

total_training_steps = len(train_loader) * EPOCHS

warmup_steps = int(0.1 * total_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)

# Mixed Precision

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),
)

# ============================================================
# Training
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

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

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(train_loader)


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0

    all_logits = []

    all_labels = []

    progress = tqdm(
        dev_loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

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

        total_loss += loss.item()

        all_logits.append(
            logits.detach().cpu().numpy()
        )

        all_labels.append(
            labels.detach().cpu().numpy()
        )

    all_logits = np.vstack(all_logits)

    all_labels = np.vstack(all_labels)

    metrics = calculate_metrics(
        all_logits,
        all_labels,
    )

    return (
        total_loss / len(dev_loader),
        metrics,
    )


# ============================================================
# Test Evaluation
# ============================================================

@torch.no_grad()
def evaluate_test():

    model.eval()

    total_loss = 0.0

    all_logits = []

    all_labels = []

    progress = tqdm(
        test_loader,
        desc="Testing",
        leave=False,
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

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

        total_loss += loss.item()

        all_logits.append(
            logits.detach().cpu().numpy()
        )

        all_labels.append(
            labels.detach().cpu().numpy()
        )

    all_logits = np.vstack(all_logits)

    all_labels = np.vstack(all_labels)

    metrics = calculate_metrics(
        all_logits,
        all_labels,
    )

    return (
        total_loss / len(test_loader),
        metrics,
    )
# ============================================================
# Training Driver
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_micro_f1 = 0.0

    patience_counter = 0

    print("=" * 70)
    print("Starting RoBERTa Fine-Tuning")
    print("=" * 70)

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch()

        val_loss, metrics = validate()

        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        print(f"Micro F1       : {metrics['micro_f1']:.4f}")
        print(f"Macro F1       : {metrics['macro_f1']:.4f}")
        print(f"Hamming Loss   : {metrics['hamming_loss']:.4f}")

        # ----------------------------------------------------
        # Save Best Model
        # ----------------------------------------------------

        if metrics["micro_f1"] > best_micro_f1:

            best_micro_f1 = metrics["micro_f1"]

            patience_counter = 0

            print("\nSaving best model...")

            model.save_pretrained(MODEL_DIR)

            tokenizer.save_pretrained(MODEL_DIR)

            shutil.copy(
                DATA_DIR / "emotions.txt",
                MODEL_DIR / "emotions.txt",
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        # ----------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

    print("\nLoading best model...")

    best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    best_model.to(DEVICE)

    global model
    model = best_model

    print("\nEvaluating on Test Set...")

    test_loss, metrics = evaluate_test()

    print("\n" + "=" * 70)
    print("Final Test Results")
    print("=" * 70)

    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Micro F1       : {metrics['micro_f1']:.4f}")
    print(f"Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"Hamming Loss   : {metrics['hamming_loss']:.4f}")

    print("\nModel saved to:")
    print(MODEL_DIR)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()