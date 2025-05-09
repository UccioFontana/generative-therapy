# Mount Google Drive to access files (datasets, models, checkpoints) from your Drive account
from google.colab import drive
drive.mount('/content/drive')  # This mounts your Drive at /content/drive/MyDrive/

# Import libraries for file handling and working with ZIP files
import zipfile
import os

# Define the path to the ZIP file containing the dataset on Google Drive
zip_path = "/content/drive/MyDrive/ai_training/processed_datasets.zip"

# Define the path where the dataset should be extracted
extract_path = "/content/drive/MyDrive/ai_training/processed_datasets"

# Check if the dataset has already been extracted to avoid redundant extraction
if not os.path.exists(extract_path):
    print("📦 Extracting dataset ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)  # Extract all contents of the ZIP file
    print("✅ Extraction completed!")
else:
    print("📂 Dataset already extracted.")  # Inform the user if dataset is already present

# Define the actual path to the HuggingFace-formatted dataset (saved using .save_to_disk)
DATASET_PATH = "/content/drive/MyDrive/ai_training/processed_datasets/processed_datasets"

# Path to save checkpoints during training (used by Hugging Face Trainer)
CHECKPOINT_PATH = "/content/drive/MyDrive/fine_tuned_model_checkpoints"

# Install necessary packages for NLP and training models using Hugging Face and scikit-learn
!pip install -q datasets transformers scikit-learn

# Import required Python libraries for training, memory management, and dataset handling
import torch
import gc
import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split

# Clear the GPU cache to free up memory before training
torch.cuda.empty_cache()
gc.collect()

# Load the dataset that was saved to disk in Hugging Face format
print(f"🔍 Loading dataset from: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)

# Display dataset structure for verification (e.g., features, size)
print("📊 Dataset structure:", dataset)

# Split the dataset into training and testing sets (80% train, 20% test)
train_size = int(len(dataset) * 0.8)
train_indices, test_indices = train_test_split(range(len(dataset)), train_size=train_size, random_state=42)
train_dataset = dataset.select(train_indices)
test_dataset = dataset.select(test_indices)

# Determine the number of unique class labels in the dataset (used to configure the model output)
NUM_LABELS = len(set(train_dataset["label"]))

# Select the best available hardware for training: CUDA (NVIDIA GPU), MPS (Apple GPU), or CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"⚙️ Using device: {device}")

# Define the base pretrained model and tokenizer to use (e.g., BERT)
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Load the tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)  # Load the model with the correct number of output labels

# Enable gradient checkpointing to save GPU memory during training (trades speed for memory)
model.gradient_checkpointing_enable()

# Check if a previous checkpoint exists, and set resume_checkpoint to the latest one if available
resume_checkpoint = None
if os.path.isdir(CHECKPOINT_PATH) and len(os.listdir(CHECKPOINT_PATH)) > 0:
    print("🔁 Checkpoint found! Resuming from the latest one...")
    # Find all folders that match the Hugging Face checkpoint naming pattern
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_PATH) if f.startswith("checkpoint-")]
    if checkpoint_files:
        # Pick the checkpoint with the highest step number (latest)
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("-")[1]))
        resume_checkpoint = os.path.join(CHECKPOINT_PATH, latest_checkpoint)
    else:
        print("⚠️ Checkpoint directory exists, but no valid checkpoint files were found.")
else:
    print("🔁 No checkpoint found, starting training from scratch.")

# Define the training configuration, including output directory, batch size, epochs, and evaluation settings
training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,                 # Directory to save checkpoints and final model
    evaluation_strategy="steps",                # Evaluate every few steps
    save_strategy="steps",                      # Save model checkpoints every few steps
    save_steps=20000,                           # Save checkpoint every 20,000 steps
    eval_steps=20000,                           # Evaluate the model every 20,000 steps
    per_device_train_batch_size=4,              # Training batch size per GPU
    per_device_eval_batch_size=4,               # Evaluation batch size per GPU
    num_train_epochs=3,                         # Number of training epochs
    weight_decay=0.01,                          # Apply weight decay to reduce overfitting
    logging_dir="./logs",                       # Directory to save training logs
    logging_steps=10,                           # Log metrics every 10 steps
    save_total_limit=3,                         # Keep only the 3 most recent checkpoints
    fp16=True if device == "cuda" else False,   # Enable mixed precision training if using CUDA
    lr_scheduler_type="linear",                 # Use a linear learning rate schedule
    warmup_steps=500,                           # Number of warmup steps for learning rate scheduler
    load_best_model_at_end=True,                # Automatically load the best model when training ends
    report_to="none",                           # Disable reporting to external services like WandB
)

# Define early stopping callback to prevent overfitting and stop training if validation doesn't improve
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

# Initialize the Hugging Face Trainer with the model, data, tokenizer, arguments, and callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping],  # Use early stopping during training
)

# Start training the model. If a checkpoint exists, resume from it
trainer.train(resume_from_checkpoint=resume_checkpoint)

# Save the final fine-tuned model and tokenizer to a specific directory on Google Drive
model.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer.save_pretrained("/content/drive/MyDrive/fine_tuned_model")

# Print a confirmation message once training and saving are complete
print("✅ Fine-tuning complete! Model saved at 'MyDrive/fine_tuned_model'")