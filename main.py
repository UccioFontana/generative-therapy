# -*- coding: utf-8 -*-
# FastAPI serving for Generative Therapy: Emotion + Dialogue
# Uses fine-tuned DeBERTa + GPT second opinion for low confidence cases

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
import os
import json
import time
import re

load_dotenv()

# --------------------------
# Configuration
# --------------------------
MODEL_DIR = os.getenv("EMO_MODEL_DIR", "BERT_sent_reg")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128

# Confidence thresholds
CONF_THRESHOLD = 0.2        # Below this → neutral
GPT_THRESHOLD = 0.45        # Below this → consult GPT
GPT_STRONG_THRESHOLD = 0.25 # Below this → GPT result overrides DeBERTa

TOPK = 5

# --------------------------
# Text cleaning
# --------------------------
def clean_utterance(text: str) -> str:
    """
    Remove any 'User (emotion: X): ' prefix from the text
    to avoid biasing the classifier.
    """
    return re.sub(r"^User\s*\(emotion:.*?\):\s*", "", text.strip(), flags=re.IGNORECASE)

# --------------------------
# Load model and labels
# --------------------------
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model not found: {MODEL_DIR}")

labels_path = os.path.join(MODEL_DIR, "labels.json")
if os.path.isfile(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        lp = json.load(f)
    EMOTION_LABELS: List[str] = lp["labels"]
    ID2LABEL: Dict[int, str] = {int(k): v for k, v in lp["id2label"].items()}
    LABEL2ID: Dict[str, int] = lp["label2id"]
else:
    EMOTION_LABELS = [
        'admiration','amusement','anger','annoyance','approval','caring','confusion',
        'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
        'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
        'pride','realization','relief','remorse','sadness','surprise','neutral'
    ]
    ID2LABEL = {i: l for i, l in enumerate(EMOTION_LABELS)}
    LABEL2ID = {l: i for i, l in ID2LABEL.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=len(EMOTION_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
).to(DEVICE)
model.eval()

# --------------------------
# OpenAI client
# --------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# Scene mapping
# --------------------------
SCENE_MAP = {
    "sadness": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "disappointment": "sadness",

    "joy": "joy",
    "gratitude": "joy",
    "love": "joy",
    "pride": "joy",
    "relief": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "optimism": "joy",
    "admiration": "joy",
    "approval": "joy",

    "anger": "anger",
    "annoyance": "anger",
    "disgust": "anger",
    "disapproval": "anger",
}
def label_to_scene(label: str) -> str:
    return SCENE_MAP.get(label, "neutral")

# --------------------------
# Minimal safety filter
# --------------------------
RISK_KEYWORDS = {
    "suicide", "kill myself", "end my life", "harm myself",
    "i want to die", "i will kill myself"
}
def safety_flag(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in RISK_KEYWORDS)

# --------------------------
# Pydantic schemas
# --------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class InputData(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

# --------------------------
# Emotion detection with DeBERTa
# --------------------------
@torch.inference_mode()
def detect_emotion(text: str) -> Dict[str, Any]:
    print(f"[DEBUG] Cleaned text for classification: '{text}'")
    t0 = time.time()
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH
    ).to(DEVICE)
    logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0]
    conf, pred_id = torch.max(probs, dim=-1)
    label = ID2LABEL[int(pred_id)]
    conf_val = float(conf.item())

    topk_conf, topk_ids = torch.topk(probs, k=min(TOPK, len(EMOTION_LABELS)))
    topk = [(ID2LABEL[int(i)], float(c)) for c, i in zip(topk_conf.tolist(), topk_ids.tolist())]

    latency_ms = int((time.time() - t0) * 1000)
    return {
        "label": label,
        "confidence": conf_val,
        "topk": topk,
        "probs": {ID2LABEL[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))},
        "latency_ms": latency_ms
    }

# --------------------------
# GPT second opinion
# --------------------------
def gpt_classify(text: str) -> str:
    try:
        prompt = (
            "You are an emotion classification model. "
            f"Classify the following text into one of these emotions: {', '.join(EMOTION_LABELS)}.\n"
            "Answer ONLY with the emotion label.\n\n"
            f"Text: {text}"
        )
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise emotion classifier."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        gpt_label = completion.choices[0].message.content.strip()
        gpt_label = gpt_label.lower().strip()
        for lbl in EMOTION_LABELS:
            if lbl.lower() == gpt_label:
                return lbl
        return None
    except Exception as e:
        print(f"[GPT classify error] {e}")
        return None

# --------------------------
# System prompt helper
# --------------------------
def ensure_system(history: List[ChatMessage]) -> List[ChatMessage]:
    if not any(m.role == "system" for m in history):
        history = [ChatMessage(
            role="system",
            content=("You are a compassionate therapist. Respond briefly and kindly. "
                     "Always end your reply with a thoughtful question that encourages the user to keep talking.")
        )] + history
    return history

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="Generative Therapy API")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "num_labels": len(EMOTION_LABELS),
        "model_dir": MODEL_DIR
    }

@app.get("/labels")
def labels():
    return {"labels": EMOTION_LABELS}

# --------------------------
# Classify endpoint (with second opinion)
# --------------------------
@app.post("/classify")
def classify(payload: Dict[str, str]):
    text = payload.get("text", "")
    cleaned_text = clean_utterance(text)
    emo = detect_emotion(cleaned_text)

    raw_label = emo["label"]
    raw_conf = emo["confidence"]

    final_label = raw_label
    gpt_label = None

    if raw_conf < GPT_THRESHOLD:
        gpt_label = gpt_classify(cleaned_text)
        if gpt_label:
            if gpt_label != raw_label and raw_conf < GPT_STRONG_THRESHOLD:
                final_label = gpt_label

    if raw_conf < CONF_THRESHOLD:
        final_label = "neutral"

    return {
        "raw_label": raw_label,
        "raw_confidence": raw_conf,
        "gpt_label": gpt_label,
        "final_label": final_label,
        "scene_signal": label_to_scene(final_label),
        "topk": emo["topk"],
        "latency_ms": emo["latency_ms"],
    }

# --------------------------
# Analyze endpoint (with second opinion)
# --------------------------
@app.post("/analyze")
def analyze(input_data: InputData):
    cleaned_input = clean_utterance(input_data.message)
    history = input_data.history or []
    history = ensure_system(history)

    emo = detect_emotion(cleaned_input)
    raw_label = emo["label"]
    raw_conf = emo["confidence"]

    final_label = raw_label
    gpt_label = None

    if raw_conf < GPT_THRESHOLD:
        gpt_label = gpt_classify(cleaned_input)
        if gpt_label:
            if gpt_label != raw_label and raw_conf < GPT_STRONG_THRESHOLD:
                final_label = gpt_label

    if raw_conf < CONF_THRESHOLD:
        final_label = "neutral"

    scene = label_to_scene(final_label)

    if safety_flag(cleaned_input):
        response_text = (
            "I'm sorry you're going through such a difficult time. "
            "You are not alone. If you are in immediate danger, please contact emergency services "
            "or a helpline in your area right now. Would you like to tell me how you are feeling at the moment?"
        )
        updated_history = history + [
            ChatMessage(role="user", content=f"User (emotion: {final_label}, conf={raw_conf:.2f}): {cleaned_input}"),
            ChatMessage(role="assistant", content=response_text),
        ]
        return {
            "raw_label": raw_label,
            "raw_confidence": raw_conf,
            "gpt_label": gpt_label,
            "final_label": final_label,
            "scene_signal": scene,
            "response": response_text,
            "updated_history": [m.dict() for m in updated_history],
            "topk": emo["topk"],
            "latency_ms": emo["latency_ms"],
            "safety": True
        }

    history.append(ChatMessage(role="user", content=f"User (emotion: {final_label}): {cleaned_input}"))

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[m.dict() for m in history],
            max_tokens=128,
            temperature=0.7,
            top_p=0.9,
        )
        response_text = completion.choices[0].message.content
        history.append(ChatMessage(role="assistant", content=response_text))
    except Exception as e:
        response_text = f"[Model error]: {e}"
        history.append(ChatMessage(role="assistant", content=response_text))

    return {
        "raw_label": raw_label,
        "raw_confidence": raw_conf,
        "gpt_label": gpt_label,
        "final_label": final_label,
        "scene_signal": scene,
        "response": response_text,
        "updated_history": [m.dict() for m in history],
        "topk": emo["topk"],
        "latency_ms": emo["latency_ms"],
        "safety": False
    }
