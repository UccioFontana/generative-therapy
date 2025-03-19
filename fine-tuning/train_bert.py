import torch  # Importing PyTorch library for model training and tensor operations
import gc  # Importing the garbage collection module to manage memory
import os  # Importing the OS library to interact with the file system
from datasets import load_from_disk  # Importing the function to load datasets from disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback  # Importing components for tokenization, model, training, and callbacks
from sklearn.model_selection import train_test_split  # Importing train_test_split for dataset splitting

# 🔥 Clear the GPU cache to avoid memory errors
torch.cuda.empty_cache()  # Clears unused GPU memory to avoid running out of memory
gc.collect()  # Invokes garbage collection to free up memory

# 📌 Load the preprocessed dataset
DATASET_PATH = os.path.abspath("../generative-therapy/processed_datasets")  # Get the absolute path to the preprocessed dataset
dataset = load_from_disk(DATASET_PATH)  # Load the dataset from the disk

# 🔍 Check the structure of the dataset
print("Dataset structure:", dataset)  # Prints the structure of the dataset to understand its format

# 📌 Split the dataset into training and testing sets (80% - 20%)
train_size = int(len(dataset) * 0.8)  # Defines the size of the training set as 80% of the dataset
train_indices, test_indices = train_test_split(range(len(dataset)), train_size=train_size, random_state=42)  # Splits the indices into training and testing

train_dataset = dataset.select(train_indices)  # Selects the training dataset using the generated indices
test_dataset = dataset.select(test_indices)  # Selects the testing dataset using the generated indices

# 📌 Determine the number of classes
NUM_LABELS = len(set(train_dataset["label"]))  # Determines the number of unique labels (classes) in the training set

# 📌 Check whether to use CUDA, MPS, or CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # Sets the device to CUDA (GPU), MPS (Apple’s GPU), or CPU based on availability

# 📌 Load the pre-trained model
MODEL_NAME = "bert-base-uncased"  # The name of the pre-trained BERT model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Loads the tokenizer for the pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)  # Loads the pre-trained BERT model for sequence classification and moves it to the selected device

# 🔥 Enable gradient checkpointing to reduce memory consumption
model.gradient_checkpointing_enable()  # Enables gradient checkpointing, a technique that saves memory during backpropagation

# 📌 Set the training arguments with smaller batch sizes and mixed precision
training_args = TrainingArguments(
    output_dir="./results",  # Directory where the model and logs will be saved
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    save_strategy="epoch",  # Save the model at the end of each epoch
    per_device_train_batch_size=4,  # Set the batch size for training to 4 (reduced to save memory)
    per_device_eval_batch_size=4,  # Set the batch size for evaluation to 4 (reduced to save memory)
    num_train_epochs=3,  # Set the number of training epochs to 3 (reduced to speed up training)
    weight_decay=0.01,  # Set weight decay for regularization
    logging_dir="./logs",  # Directory for logging training information
    logging_steps=10,  # Log training details every 10 steps
    save_total_limit=2,  # Keep only the last 2 saved models to save disk space
    fp16=True if device == "cuda" else False,  # Enable mixed precision (FP16) training only if a CUDA device is used
    lr_scheduler_type="linear",  # Use a linear learning rate scheduler
    warmup_steps=500,  # Number of warm-up steps to gradually increase the learning rate
    load_best_model_at_end=True,  # Load the best model after training is complete
)

# 📌 Add Early Stopping callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)  # Stop training if the model does not improve after 2 epochs

# 📌 Create the Trainer
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # The training dataset
    eval_dataset=test_dataset,  # The testing dataset
    tokenizer=tokenizer,  # The tokenizer used for preprocessing the text
    callbacks=[early_stopping]  # Add the EarlyStoppingCallback to the trainer
)

# 🚀 Start the training
trainer.train()  # Starts the training process

# 📌 Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")  # Save the trained model to the specified directory
tokenizer.save_pretrained("./fine_tuned_model")  # Save the tokenizer to the specified directory

print("✅ Fine-tuning completed! The model is saved in './fine_tuned_model'")  # Prints a completion message indicating the model has been saved