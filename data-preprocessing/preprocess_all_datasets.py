# -*- coding: utf-8 -*-
"""
Preprocessing coerente con la tesi:
- Tassonomia unica: 28 classi (27 GoEmotions + 'neutral')
- GoEmotions robusto agli split (train/validation/test o solo train)
- DailyDialog (mirror OpenRL/daily_dialog) rimappato su 28, con schema robusto
- Reddit MH rietichettato con weak labeling (teacher: SamLowe/roberta-base-go_emotions)
- PII cleaning (email, telefono, @username, URL) con conteggio redazioni
- Tokenizzazione max_length=128
- Bilanciamento sorgenti e interleave
- Split stratificato (sklearn)
- Salvataggio labels.json, split_meta.json, dataset su disco
"""

import json
import os
import re
from typing import Dict, Tuple, List

import numpy as np
from datasets import (
    load_dataset, interleave_datasets, Value, concatenate_datasets, Dataset, DatasetDict
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # Deve dare 'NVIDIA GeForce RTX 2080'



# -----------------------
# Config
# -----------------------
MODEL_NAME = "bert-base-uncased"   # tokenizer di riferimento
MAX_LENGTH = 128                   # standard per training/inferenza
OUT_DIR = "processed_datasets"     # cartella output

USE_REDDIT = True                  # True per includere Reddit MH (weak labeling)
REDDIT_TEACHER = "SamLowe/roberta-base-go_emotions"  # teacher pubblico e stabile
REDDIT_CONF_THRESHOLD = 0.5        # soglia minima di confidenza (0.6-0.7 = più pulito, meno dati)

SEED = 42
TEST_SIZE = 0.2

# Mirror stabile di DailyDialog (lo "storico" può risultare deprecato/rotto)
DD_DATASET_NAME = "OpenRL/daily_dialog"

os.makedirs(OUT_DIR, exist_ok=True)

# Tassonomia unica (ordine fisso) — 27 GoEmotions + 'neutral'
EMOTION_LABELS: List[str] = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(EMOTION_LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}

# DailyDialog mapping (0..6) -> nostre 28 etichette
# 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happy, 5: sadness, 6: surprise
DD_INT_TO_LABEL = {
    0: 'neutral',
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'joy',
    5: 'sadness',
    6: 'surprise'
}

# Tokenizer principale
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------
# PII cleaning
# -----------------------
PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '[REDACTED_EMAIL]'),
    (re.compile(r'(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4})'), '[REDACTED_PHONE]'),
    (re.compile(r'@\w+'), '[REDACTED_USER]'),
    (re.compile(r'https?://\S+|www\.\S+'), '[REDACTED_URL]')
]

def clean_pii(text: str) -> Tuple[str, int]:
    if not isinstance(text, str):
        return "", 0
    redactions = 0
    cleaned = text
    for pat, repl in PII_PATTERNS:
        cleaned, n = pat.subn(repl, cleaned)
        redactions += n
    return cleaned.strip(), redactions

# -----------------------
# GoEmotions
# -----------------------
def preprocess_goemotions(ds_split: Dataset) -> Dataset:
    """
    Config 'raw': 27 colonne binarie (senza 'neutral') + text.
    Multi-label -> single-label: se c'è almeno 1, prendi argmax; altrimenti 'neutral'.
    """
    emo_cols = EMOTION_LABELS[:-1]  # tutte tranne 'neutral'
    neutral_idx = LABEL2ID['neutral']

    def map_row(example):
        values = [int(example.get(lbl, 0)) for lbl in emo_cols]
        if sum(values) > 0:
            label_idx = int(np.argmax(values))
            label = emo_cols[label_idx]
            y = LABEL2ID[label]
        else:
            y = neutral_idx
        text = example["text"] if "text" in example else ""
        text, red = clean_pii(text)
        return {"text": text, "label": int(y), "pii_redactions": int(red)}

    ds = ds_split.map(map_row, remove_columns=[c for c in ds_split.column_names if c not in ("text",)])
    ds = ds.cast_column("label", Value("int64"))
    ds = ds.cast_column("text", Value("string"))
    ds = ds.cast_column("pii_redactions", Value("int64"))
    return ds

# -----------------------
# DailyDialog (robusto allo schema del mirror OpenRL)
# -----------------------
def preprocess_dailydialog(ds_split: Dataset) -> Dataset:
    """
    Supporta varianti del dataset (es. OpenRL/daily_dialog):
    - 'dialog' può essere lista di turni o stringa
    - 'emotion' può essere lista di interi (con -1 come 'no emotion') o mancare
    - Alcuni fork usano 'utterances', 'label' o 'labels'
    - In assenza di etichette valide -> 'neutral'
    """
    keep_cols = [c for c in ds_split.column_names if c in ("dialog", "emotion", "utterances", "label", "labels")]
    ds = ds_split.remove_columns([c for c in ds_split.column_names if c not in keep_cols])

    def to_text_and_label(x):
        # --- testo ---
        if "dialog" in x:
            d = x["dialog"]
        elif "utterances" in x:
            d = x["utterances"]
        else:
            d = None

        if isinstance(d, list):
            text = " ".join(u for u in d if isinstance(u, str))
        elif isinstance(d, str):
            text = d
        else:
            text = ""

        # --- emozioni grezze ---
        if "emotion" in x:
            raw_emotions = x["emotion"]
        elif "label" in x:
            raw_emotions = x["label"]
        elif "labels" in x:
            raw_emotions = x["labels"]
        else:
            raw_emotions = None

        labels = []
        if isinstance(raw_emotions, list):
            # tipicamente lista di int 0..6 o -1
            for el in raw_emotions:
                if isinstance(el, (int, np.integer)) and el >= 0:
                    labels.append(int(el))
        elif isinstance(raw_emotions, (int, np.integer)) and raw_emotions >= 0:
            labels = [int(raw_emotions)]
        # altrimenti: nessuna etichetta valida -> neutral

        # moda delle emozioni (0..6), mappa su nostra tassonomia
        if len(labels) > 0:
            vals, counts = np.unique(labels, return_counts=True)
            label_dd = int(vals[np.argmax(counts)])
            label_name = DD_INT_TO_LABEL.get(label_dd, "neutral")
        else:
            label_name = "neutral"

        y = LABEL2ID[label_name]

        # PII cleaning
        text, red = clean_pii(text)

        return {"text": text, "label": int(y), "pii_redactions": int(red)}

    ds = ds.map(to_text_and_label)
    ds = ds.cast_column("label", Value("int64"))
    ds = ds.cast_column("text", Value("string"))
    ds = ds.cast_column("pii_redactions", Value("int64"))
    return ds

# -----------------------
# Reddit MH (weak labeling con teacher multi-label)
# -----------------------
def preprocess_reddit_with_teacher(
    ds_split: Dataset,
    teacher_model_name: str = REDDIT_TEACHER,
    max_length: int = MAX_LENGTH,
    confidence_threshold: float = REDDIT_CONF_THRESHOLD,
    device: str | None = None,
    batch_size: int = 128
) -> Dataset:
    """
    Reddit MH → weak labeling con teacher (multi-label GoEmotions).
    Costruzione manuale del dataset (senza map batched) per permettere il filtraggio.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_model_name).to(device)
    teacher.eval()

    # id2label del teacher
    teacher_id2label = getattr(teacher.config, "id2label", {i: l for i, l in enumerate(EMOTION_LABELS)})

    # Accumulatori
    out_text, out_label, out_red = [], [], []

    def infer_batch(texts):
        enc = teacher_tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = teacher(**enc).logits    # [B, 28]
            probs = torch.sigmoid(logits)     # multi-label
            confs, preds = torch.max(probs, dim=-1)
        return preds.detach().cpu().tolist(), confs.detach().cpu().tolist()

    # Estrai colonne necessarie
    cols_keep = [c for c in ds_split.column_names if c in ("id", "body")]
    ds = ds_split.remove_columns([c for c in ds_split.column_names if c not in cols_keep])

    n = len(ds)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ds.select(range(start, end))
        bodies = batch["body"]

        cleaned, reds = [], []
        for t in bodies:
            t2, r = clean_pii(t)
            cleaned.append(t2)
            reds.append(r)

        preds, confs = infer_batch(cleaned)

        for txt, pred_idx, conf, rcount in zip(cleaned, preds, confs, reds):
            label_name = teacher_id2label.get(int(pred_idx), "neutral")
            if label_name not in LABEL2ID:
                label_name = "neutral"
            if float(conf) >= confidence_threshold:
                out_text.append(txt)
                out_label.append(int(LABEL2ID[label_name]))
                out_red.append(int(rcount))

    print(f"🧪 Reddit MH (weak): tenuti {len(out_text)} esempi con confidenza >= {confidence_threshold}")

    if len(out_text) == 0:
        # Ritorna un dataset vuoto compatibile
        return Dataset.from_dict({"text": [], "label": [], "pii_redactions": []}).cast_column("label", Value("int64"))

    ds_labeled = Dataset.from_dict({
        "text": out_text,
        "label": out_label,
        "pii_redactions": out_red
    })
    ds_labeled = ds_labeled.cast_column("label", Value("int64"))
    ds_labeled = ds_labeled.cast_column("text", Value("string"))
    ds_labeled = ds_labeled.cast_column("pii_redactions", Value("int64"))
    return ds_labeled

# -----------------------
# Tokenizzazione
# -----------------------
def tokenize_dataset(ds: Dataset) -> Dataset:
    def tok(batch):
        texts = batch["text"]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    cols_to_remove = [c for c in ds.column_names if c not in ("text", "label", "pii_redactions")]
    ds = ds.map(tok, batched=True, remove_columns=cols_to_remove)
    return ds

# -----------------------
# Oversampling / match size
# -----------------------
def oversample_to_match(ds_small: Dataset, target_size: int, seed: int = SEED) -> Dataset:
    if len(ds_small) == target_size:
        return ds_small
    repeat = target_size // len(ds_small)
    extra = target_size % len(ds_small)
    parts = [ds_small] * repeat
    if extra > 0:
        parts.append(ds_small.shuffle(seed=seed).select(range(extra)))
    return concatenate_datasets(parts)

def oversample_to_match_any(ds: Dataset, target: int, seed: int = SEED) -> Dataset:
    if len(ds) < target:
        return oversample_to_match(ds, target, seed=seed)
    elif len(ds) > target:
        return ds.select(range(target))
    return ds

# -----------------------
# Split stratificato (sklearn)
# -----------------------
def stratified_split(ds: Dataset, test_size: float = TEST_SIZE, seed: int = SEED) -> DatasetDict:
    labels = ds["label"]
    idx = np.arange(len(labels))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(idx, labels))
    ds_train = ds.select(train_idx.tolist())
    ds_test  = ds.select(test_idx.tolist())
    return DatasetDict({"train": ds_train, "test": ds_test})

# -----------------------
# Main
# -----------------------
def main():
    print("🔽 Carico GoEmotions (raw) e DailyDialog (mirror OpenRL)…")
    goemotions = load_dataset("go_emotions", "raw")
    print(f"   GoEmotions splits disponibili: {list(goemotions.keys())}")

    geo_parts = []
    if "train" in goemotions:
        geo_parts.append(preprocess_goemotions(goemotions["train"]))
    if "validation" in goemotions:
        geo_parts.append(preprocess_goemotions(goemotions["validation"]))
    if "test" in goemotions:
        geo_parts.append(preprocess_goemotions(goemotions["test"]))

    if len(geo_parts) == 0:
        raise RuntimeError("GoEmotions non ha split disponibili.")
    elif len(geo_parts) == 1:
        geo_full = geo_parts[0]
    else:
        geo_full = concatenate_datasets(geo_parts)

    # DailyDialog dal mirror OpenRL
    dailydialog = load_dataset(DD_DATASET_NAME)
    print(f"   DailyDialog({DD_DATASET_NAME}) splits: {list(dailydialog.keys())}")
    # Log utile per diagnosticare eventuali futuri cambi schema
    try:
        print("   Colonne DailyDialog(train):", dailydialog["train"].column_names)
    except Exception:
        pass

    dd_parts = []
    if "train" in dailydialog:
        dd_parts.append(preprocess_dailydialog(dailydialog["train"]))
    if "validation" in dailydialog:
        dd_parts.append(preprocess_dailydialog(dailydialog["validation"]))
    if "test" in dailydialog:
        dd_parts.append(preprocess_dailydialog(dailydialog["test"]))

    if len(dd_parts) == 0:
        raise RuntimeError("DailyDialog non ha split disponibili.")
    elif len(dd_parts) == 1:
        dd_full = dd_parts[0]
    else:
        dd_full = concatenate_datasets(dd_parts)

    # Reddit MH opzionale con weak labeling
    reddit_bal = None
    if USE_REDDIT:
        try:
            print("🔽 Carico Reddit Mental Health…")
            reddit_ds = load_dataset("solomonk/reddit_mental_health_posts")
            reddit_weak = preprocess_reddit_with_teacher(
                reddit_ds["train"],
                teacher_model_name=REDDIT_TEACHER,
                max_length=MAX_LENGTH,
                confidence_threshold=REDDIT_CONF_THRESHOLD
            )
            print("⚖️  Bilancio dimensioni tra le sorgenti…")
            max_size = max(len(geo_full), len(dd_full), len(reddit_weak))
            geo_full_bal = oversample_to_match_any(geo_full, max_size)
            dd_full_bal  = oversample_to_match_any(dd_full,  max_size)
            reddit_bal   = oversample_to_match_any(reddit_weak, max_size)
            print(f"   -> GoE={len(geo_full_bal)}, DD={len(dd_full_bal)}, Reddit={len(reddit_bal)}")
            sources = [geo_full_bal, dd_full_bal, reddit_bal]
        except Exception as e:
            print(f"⚠️  Reddit MH non disponibile o errore nel weak labeling: {e}")
            print("   Procedo con GoEmotions + DailyDialog.")
            max_size = max(len(geo_full), len(dd_full))
            geo_full_bal = oversample_to_match_any(geo_full, max_size)
            dd_full_bal  = oversample_to_match_any(dd_full,  max_size)
            sources = [geo_full_bal, dd_full_bal]
    else:
        print("ℹ️  Reddit MH escluso dal preprocessing.")
        max_size = max(len(geo_full), len(dd_full))
        geo_full_bal = oversample_to_match_any(geo_full, max_size)
        dd_full_bal  = oversample_to_match_any(dd_full,  max_size)
        sources = [geo_full_bal, dd_full_bal]

    print("🔀 Interleave datasets…")
    combined = interleave_datasets(sources, seed=SEED)

    print("🔤 Tokenizzo (max_length=128)…")
    tokenized = tokenize_dataset(combined)

    print("✂️  Split stratificato train/test…")
    dsdict: DatasetDict = stratified_split(tokenized, test_size=TEST_SIZE, seed=SEED)

    # Metadati split per riproducibilità
    split_meta = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "stratify_by": "label",
        "sources": {
            "go_emotions": len(geo_full),
            "daily_dialog": len(dd_full),
            "reddit_weak" : None if reddit_bal is None else len(reddit_bal)
        },
        "note": "Split riproducibile con stessi dati, seed e pipeline."
    }
    with open(os.path.join(OUT_DIR, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)

    # Label mapping (id2label/label2id) per training e backend
    labels_payload = {
        "labels": EMOTION_LABELS,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()}
    }
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2, ensure_ascii=False)

    # Salvataggio dataset
    print(f"💾 Salvo dataset su: {OUT_DIR}")
    dsdict.save_to_disk(OUT_DIR)

    # Riepilogo PII redactions
    red_train = int(np.sum(dsdict["train"]["pii_redactions"]))
    red_test  = int(np.sum(dsdict["test"]["pii_redactions"]))
    print("✅ Preprocessing completo!")
    print(f"   - Train size: {len(dsdict['train'])} | Test size: {len(dsdict['test'])}")
    print(f"   - PII redactions -> train: {red_train} | test: {red_test}")
    print(f"   - Labels salvate in {os.path.join(OUT_DIR, 'labels.json')}")
    if USE_REDDIT:
        kept = split_meta["sources"]["reddit_weak"]
        print(f"   - Reddit MH (weak) incluso: {kept if kept is not None else 0} esempi")

if __name__ == "__main__":
    main()
