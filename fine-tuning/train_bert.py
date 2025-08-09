# -*- coding: utf-8 -*-
"""
Training DeBERTa-v3-base senza 'neutral' + class weights (fix per Windows):
- Filtra 'neutral' da train/val/test con num_proc=1
- Tokenizza al volo se mancano input_ids/attention_mask
- Mantiene solo {input_ids, attention_mask, labels}
- Usa default_data_collator (batch già pad lato tokenizzazione)
- Disabilita multiprocessing nei DataLoader
"""

import os
import json
import math
import numpy as np
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    default_data_collator,
)

# =========================
# Config
# =========================
DATASET_DIR = "processed_datasets"
OUTPUT_DIR  = "DEBERTA_sent_reg_no_neutral"
BASE_MODEL  = "microsoft/deberta-v3-base"
SEED        = 42

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL  = 16
NUM_EPOCHS       = 4
LEARNING_RATE    = 2e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
LOG_STEPS        = 50
EVAL_STEPS       = 5000
SAVE_STEPS       = 5000
EARLY_STOP_PATIENCE = 5
MAX_CLASS_WEIGHT = 4.0
MAX_LENGTH       = 128

# =========================
# Device + seed
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")
set_seed(SEED)
torch.backends.cudnn.benchmark = True if device == "cuda" else False

# =========================
# Dataset + labels
# =========================
if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(f"Dataset non trovato: {DATASET_DIR}")

dsdict: DatasetDict = load_from_disk(DATASET_DIR)
if not {"train", "test"}.issubset(dsdict.keys()):
    raise RuntimeError(f"Split 'train' e 'test' mancanti in {DATASET_DIR}")

labels_path = os.path.join(DATASET_DIR, "labels.json")
with open(labels_path, "r", encoding="utf-8") as f:
    labels_payload = json.load(f)

all_labels: List[str] = labels_payload["labels"]

# Filtra 'neutral'
if "neutral" not in all_labels:
    raise ValueError("'neutral' non trovato nelle label")
print("❌ Rimuovo 'neutral' dal training/test...")
non_neutral_labels = [lbl for lbl in all_labels if lbl != "neutral"]

LABEL2ID = {lbl: i for i, lbl in enumerate(non_neutral_labels)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
num_labels = len(non_neutral_labels)
print(f"🔖 Classi usate: {num_labels} -> {list(non_neutral_labels)}")

neutral_id = labels_payload["label2id"]["neutral"]

def filter_neutral(ex):
    return ex["label"] != neutral_id

# FIX Windows: num_proc=1
train_full: Dataset = dsdict["train"].filter(filter_neutral, num_proc=1)
test_non_neutral: Dataset = dsdict["test"].filter(filter_neutral, num_proc=1)

# =========================
# Split stratificato per val
# =========================
y = [LABEL2ID[all_labels[lbl]] for lbl in train_full["label"]]
idx = np.arange(len(y))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(sss.split(idx, y))
train_ds = train_full.select(train_idx.tolist())
val_ds   = train_full.select(val_idx.tolist())

# =========================
# Remap labels e rinomina in 'labels'
# =========================
def remap_labels(example):
    example["label"] = LABEL2ID[all_labels[example["label"]]]
    return example

train_ds = train_ds.map(remap_labels).rename_column("label", "labels")
val_ds   = val_ds.map(remap_labels).rename_column("label", "labels")
test_non_neutral = test_non_neutral.map(remap_labels).rename_column("label", "labels")

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

TEXT_CANDIDATES = ["text", "clean_text", "utterance"]

def guess_text_field(columns):
    for c in TEXT_CANDIDATES:
        if c in columns:
            return c
    return None

def need_tokenization(ds: Dataset) -> bool:
    cols = set(ds.column_names)
    return not {"input_ids", "attention_mask"}.issubset(cols)

def tokenize_dataset(ds: Dataset, name: str) -> Dataset:
    if not need_tokenization(ds):
        print(f"✅ {name}: già tokenizzato (uso input_ids/attention_mask).")
        return ds
    text_col = guess_text_field(ds.column_names)
    if text_col is None:
        raise ValueError(
            f"{name}: mancano input_ids/attention_mask e non trovo una colonna di testo. "
            f"Colonne presenti: {ds.column_names}"
        )
    print(f"✍️  Tokenizzo {name} dal campo '{text_col}'...")
    def tok_fn(batch):
        enc = tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        return enc
    keep = ["labels", text_col]
    remove_cols = [c for c in ds.column_names if c not in keep]
    ds = ds.map(tok_fn, batched=True, remove_columns=remove_cols)
    keep2 = [c for c in ["input_ids", "attention_mask", "labels"] if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep2])
    print(f"✅ {name}: tokenizzato. Colonne: {ds.column_names}")
    return ds

train_ds = tokenize_dataset(train_ds, "train")
val_ds   = tokenize_dataset(val_ds, "val")
test_non_neutral = tokenize_dataset(test_non_neutral, "test_no_neutral")

# =========================
# Torch format
# =========================
required_columns = ["input_ids", "attention_mask", "labels"]
for ds_name, ds_obj in [("train", train_ds), ("val", val_ds), ("test_no_neutral", test_non_neutral)]:
    missing = [c for c in required_columns if c not in ds_obj.column_names]
    if missing:
        raise ValueError(f"{ds_name}: mancano colonne richieste {missing}, trovate {ds_obj.column_names}")
    ds_obj.set_format(type="torch", columns=required_columns)

# =========================
# Class weights
# =========================
labels_array = np.array(train_ds["labels"])
class_counts = np.bincount(labels_array, minlength=num_labels)
class_weights = 1.0 / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.sum() * num_labels
class_weights = np.minimum(class_weights, MAX_CLASS_WEIGHT)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print("⚖️  Class weights (capped):", class_weights)

# =========================
# Modello con pesi nella loss
# =========================
class WeightedModel(nn.Module):
    def __init__(self, base_model_name, num_labels, id2label, label2id, class_weights):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if labels is not None:
            loss = self.loss_fct(outputs.logits, labels)
            return {"loss": loss, "logits": outputs.logits}
        return outputs

model = WeightedModel(BASE_MODEL, num_labels, ID2LABEL, LABEL2ID, class_weights_tensor).to(device)

# =========================
# Metriche
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1}

# =========================
# Training args (Windows-friendly)
# =========================
train_steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE_TRAIN)
total_train_steps = train_steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_train_steps * WARMUP_RATIO)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    logging_steps=LOG_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none",
    fp16=(device == "cuda"),
    dataloader_pin_memory=True,
    dataloader_num_workers=0  # FIX Windows
)

# =========================
# Trainer
# =========================
callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)]

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    callbacks=callbacks
)

# =========================
# Train
# =========================
print(f"🚀 Training: {len(train_ds)} train, {len(val_ds)} val, {len(test_non_neutral)} test (NO neutral)")
train_result = trainer.train()

# =========================
# Test finale
# =========================
print("🧪 Test finale…")
test_metrics = trainer.evaluate(test_non_neutral, metric_key_prefix="test")
print(test_metrics)

# =========================
# Save
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump({
        "labels": non_neutral_labels,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()}
    }, f, indent=2, ensure_ascii=False)

print("✅ Training completato.")
print("   Metriche test:", test_metrics)
