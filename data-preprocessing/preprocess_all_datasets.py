from datasets import load_dataset, interleave_datasets, Value, concatenate_datasets
import numpy as np
from transformers import AutoTokenizer

# Load a pre-trained tokenizer from the Hugging Face Transformers library.
# This tokenizer will be used to convert text into token IDs for model input.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Choose the appropriate model

# Load multiple text-based datasets from the Hugging Face datasets library.
goemotions = load_dataset("go_emotions", "raw")  # GoEmotions dataset

dailydialog = load_dataset("daily_dialog")  # DailyDialog dataset

reddit_mental_health = load_dataset("solomonk/reddit_mental_health_posts")  # Reddit Mental Health dataset

# Function to preprocess the GoEmotions dataset
def preprocess_goemotions(dataset):
    # Define the emotion labels used in the dataset.
    emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                      'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                      'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                      'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

    def map_emotions(example):
        # Convert multi-label emotion annotations to a single-label classification by selecting the most intense emotion.
        emotion_values = [example[label] for label in emotion_labels]
        example["label"] = np.argmax(emotion_values) if sum(emotion_values) > 0 else 27  # Default to 'neutral' if no emotion is present
        return {"text": example["text"], "label": int(example["label"])}

    # Apply the transformation to the entire dataset.
    dataset = dataset.map(map_emotions)
    
    # Remove unnecessary columns from the dataset to keep only 'text' and 'label'.
    dataset = dataset.remove_columns(emotion_labels + ["id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"])
    
    return dataset

# Function to preprocess the DailyDialog dataset
def preprocess_dailydialog(dataset):
    # Rename the 'emotion' column to 'label' for consistency across datasets.
    dataset = dataset.rename_column("emotion", "label")

    def process_example(x):
        # Combine multiple utterances in a dialog into a single string.
        text = " ".join(x["dialog"])
        # Extract the first emotion label (if available) or default to 0.
        label = int(x["label"][0]) if isinstance(x["label"], list) and len(x["label"]) > 0 else 0
        return {"text": text, "label": label}

    # Apply transformations to the dataset.
    dataset = dataset.map(process_example)
    
    # Remove unnecessary columns to keep only 'text' and 'label'.
    dataset = dataset.remove_columns(["dialog", "act"])
    return dataset

# Function to preprocess the Reddit Mental Health dataset
def preprocess_reddit(dataset):
    if "subreddit" in dataset.column_names:
        # Create a mapping of subreddit names to numerical labels.
        unique_labels = list(set(dataset["subreddit"]))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        dataset = dataset.map(lambda x: {"label": int(label_mapping[x["subreddit"]])})
        
        # Rename 'body' to 'text' for consistency across datasets.
        dataset = dataset.rename_column("body", "text")
        
        # Remove unnecessary columns.
        dataset = dataset.remove_columns(["id", "author", "title", "created_utc", "num_comments", "score", "upvote_ratio", "url"])
    else:
        raise ValueError("No valid label column found in the Reddit dataset!")
    return dataset

# Ensure correct data types for all datasets
def ensure_correct_types(dataset):
    dataset = dataset.cast_column("label", Value("int64"))  # Ensure labels are integers
    dataset = dataset.cast_column("text", Value("string"))  # Ensure text is stored as strings
    return dataset

# Function to tokenize text data
def tokenize_dataset(dataset):
    def tokenize_function(examples):
        # Ensure the 'text' column exists
        if "text" not in examples:
            raise ValueError("❌ Error: 'text' column is missing in the dataset!")
        
        texts = [t if isinstance(t, str) else "" for t in examples["text"]]  # Handle empty values
        
        # Tokenize text data with padding and truncation
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        return {"input_ids": encodings["input_ids"].tolist(), "attention_mask": encodings["attention_mask"].tolist()}

    dataset = dataset.map(tokenize_function, batched=True)  # Apply tokenization in batches
    return dataset

# Apply preprocessing steps to all datasets
goemotions_processed = preprocess_goemotions(goemotions["train"])
dailydialog_processed = preprocess_dailydialog(dailydialog["train"])
reddit_processed = preprocess_reddit(reddit_mental_health["train"])

# Ensure consistent data types
goemotions_processed = ensure_correct_types(goemotions_processed)
dailydialog_processed = ensure_correct_types(dailydialog_processed)
reddit_processed = ensure_correct_types(reddit_processed)

# Apply tokenization to all datasets
goemotions_processed = tokenize_dataset(goemotions_processed)
dailydialog_processed = tokenize_dataset(dailydialog_processed)
reddit_processed = tokenize_dataset(reddit_processed)

# Oversample datasets to match the size of the largest dataset
max_size = max(len(goemotions_processed), len(dailydialog_processed), len(reddit_processed))

def oversample(dataset, target_size):
    repeat_factor = target_size // len(dataset)
    extra_samples = target_size % len(dataset)
    dataset_repeated = concatenate_datasets([dataset] * repeat_factor)
    dataset_extra = dataset.shuffle(seed=42).select(range(extra_samples))
    return concatenate_datasets([dataset_repeated, dataset_extra])

# Apply oversampling to balance dataset sizes
goemotions_balanced = oversample(goemotions_processed, max_size)
dailydialog_balanced = oversample(dailydialog_processed, max_size)
reddit_balanced = oversample(reddit_processed, max_size)

# Interleave datasets to create a balanced dataset with equal representation
balanced_dataset = interleave_datasets([goemotions_balanced, dailydialog_balanced, reddit_balanced], seed=42)

# Save the final dataset to disk
balanced_dataset.save_to_disk("processed_datasets")

print("✅ Dataset preprocessing and tokenization complete! Saved to 'processed_datasets'.")
