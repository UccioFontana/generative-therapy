# 📌 MONTA GOOGLE DRIVE (se vuoi salvare il modello lì)
from google.colab import drive
drive.mount('/content/drive')

# 🔓 ESTRAI IL FILE ZIP SE NECESSARIO
import zipfile
import os

zip_path = "/content/drive/MyDrive/ai_training/processed_datasets.zip"
extract_path = "/content/drive/MyDrive/ai_training/processed_datasets"

if not os.path.exists(extract_path):
    print("📦 Estrazione del dataset ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Estrazione completata!")
else:
    print("📂 Dataset già estratto.")

# ✅ IMPOSTA IL PERCORSO DEL DATASET (modifica se necessario)
DATASET_PATH = "/content/drive/MyDrive/ai_training/processed_datasets/processed_datasets"

# ✅ PERCORSO CHECKPOINT SU GOOGLE DRIVE
CHECKPOINT_PATH = "/content/drive/MyDrive/fine_tuned_model_checkpoints"

# 📦 INSTALLA I PACCHETTI NECESSARI
!pip install -q datasets transformers scikit-learn

# 📥 IMPORTAZIONI
import torch
import gc
import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split

# 🔥 LIBERA LA CACHE GPU
torch.cuda.empty_cache()
gc.collect()

# 📌 CARICA IL DATASET
print(f"🔍 Caricamento del dataset da: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)

# 👀 VERIFICA STRUTTURA
print("📊 Struttura del dataset:", dataset)

# 📌 SPLIT: 80% train, 20% test
train_size = int(len(dataset) * 0.8)
train_indices, test_indices = train_test_split(range(len(dataset)), train_size=train_size, random_state=42)
train_dataset = dataset.select(train_indices)
test_dataset = dataset.select(test_indices)

# 📌 CALCOLA IL NUMERO DI CLASSI
NUM_LABELS = len(set(train_dataset["label"]))

# 💻 SCEGLI IL DISPOSITIVO
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"⚙️ Usando dispositivo: {device}")

# 📌 MODELLO E TOKENIZER
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)

# 💡 ATTIVA GRADIENT CHECKPOINTING PER RIDURRE USO RAM GPU
model.gradient_checkpointing_enable()

# ✅ CONTROLLA SE ESISTE UN CHECKPOINT
resume_checkpoint = None
if os.path.isdir(CHECKPOINT_PATH) and len(os.listdir(CHECKPOINT_PATH)) > 0:
    print("🔁 Checkpoint trovato! Riprendo da lì...")
    # Find the latest checkpoint file
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_PATH) if f.startswith("checkpoint-")]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("-")[1]))
        resume_checkpoint = os.path.join(CHECKPOINT_PATH, latest_checkpoint)  # Use the latest checkpoint file
    else:
        print("⚠️ Checkpoint directory exists, but no checkpoint files found. Starting from scratch.")
else:
    print("🔁 Nessun checkpoint trovato, inizio da zero.")

# ⚙️ ARGOMENTI DI TRAINING
training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,  # ✅ Salva checkpoint direttamente su Drive
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20000,
    eval_steps=20000,# ✅ Ogni 500 step (modificabile)
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=3,          # ✅ Tiene solo ultimi 3 checkpoint
    fp16=True if device == "cuda" else False,
    lr_scheduler_type="linear",
    warmup_steps=500,
    load_best_model_at_end=True,
    report_to="none",
)

# 🛑 CALLBACK: EARLY STOPPING
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

# 🧠 CREAZIONE DEL TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping],
)

# 🚀 INIZIA L'ALLENAMENTO (con resume se c'è checkpoint)
trainer.train(resume_from_checkpoint=resume_checkpoint)

# 💾 SALVA MODELLO FINE-TUNED SU DRIVE
model.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer.save_pretrained("/content/drive/MyDrive/fine_tuned_model")

print("✅ Fine-tuning completato! Modello salvato in 'MyDrive/fine_tuned_model'")