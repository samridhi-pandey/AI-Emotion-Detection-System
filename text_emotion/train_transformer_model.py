
# Final train_transformer_model.py
# RoBERTa fine-tuning for GoEmotions (multi-label)

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "text_emotion" / "goemotions"
MODEL_DIR = Path(__file__).resolve().parent / "transformer_model"

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
EPOCHS = 3
LR = 2e-5
TRAIN_BS = 8
EVAL_BS = 8

def load_emotions():
    return [x.strip() for x in open(DATA_DIR/"emotions.txt",encoding="utf-8")]

def load_split(name):
    df = pd.read_csv(DATA_DIR/f"{name}.tsv",sep="\t",header=None,names=["text","labels","id"])
    df["label_list"]=df["labels"].apply(lambda s:[int(i) for i in str(s).split(",")])
    return df

def prepare(df, mlb):
    y = mlb.transform(df["label_list"]).astype("float32")
    df=df.copy()
    df["labels"]=y.tolist()
    return Dataset.from_pandas(df[["text","labels"]])

def metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1/(1+np.exp(-logits))
    preds=(probs>=0.5).astype(int)
    return {
        "micro_f1":f1_score(labels,preds,average="micro",zero_division=0),
        "macro_f1":f1_score(labels,preds,average="macro",zero_division=0),
        "hamming_loss":hamming_loss(labels,preds)
    }

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    emotions=load_emotions()

    train_df=load_split("train")
    dev_df=load_split("dev")
    test_df=load_split("test")

    mlb=MultiLabelBinarizer(classes=list(range(len(emotions))))
    mlb.fit(train_df["label_list"])

    train_ds=prepare(train_df,mlb)
    dev_ds=prepare(dev_df,mlb)
    test_ds=prepare(test_df,mlb)

    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"],truncation=True,max_length=MAX_LENGTH)

    train_ds=train_ds.map(tok,batched=True,remove_columns=["text"])
    dev_ds=dev_ds.map(tok,batched=True,remove_columns=["text"])
    test_ds=test_ds.map(tok,batched=True,remove_columns=["text"])

    train_ds.set_format("torch")
    dev_ds.set_format("torch")
    test_ds.set_format("torch")

    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(emotions),
        problem_type="multi_label_classification"
    )

    args=TrainingArguments(
        output_dir=str(MODEL_DIR/"checkpoints"),
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer=Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Training...")
    trainer.train()

    print("Testing...")
    print(trainer.evaluate(test_ds))

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(MODEL_DIR/"emotions.txt","w",encoding="utf-8") as f:
        f.write("\n".join(emotions))
    print("Saved to", MODEL_DIR)

if __name__=="__main__":
    main()