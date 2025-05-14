# --- Import necessary libraries ---
from fastapi import FastAPI  # Web framework for building APIs with Python
from pydantic import BaseModel  # Data validation and parsing using Python type annotations
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face tools for loading pretrained models
from openai import OpenAI  # OpenAI SDK to access language models
from typing import List, Optional  # Type hinting tools
import torch  # PyTorch for model inference
import os  # To access environment variables
from dotenv import load_dotenv  # Loads environment variables from a .env file

# --- Load environment variables from .env file (especially OpenAI API key) ---
load_dotenv()

# --- CONFIGURATION SECTION ---
# Select computation device: use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to the fine-tuned emotion classification model
emotion_model_path = "BERT_sent_reg"

# --- Load Emotion Detection Model ---
# Load tokenizer and model from the specified path
tokenizer_bert = AutoTokenizer.from_pretrained(emotion_model_path)
model_bert = AutoModelForSequenceClassification.from_pretrained(emotion_model_path).to(device)

# Define the possible emotion labels the model can predict
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# --- Initialize OpenAI client using the loaded API key ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Initialize FastAPI app instance ---
app = FastAPI()

# --- Define Pydantic data models for request validation ---

# Represents a single message in a chat history
class ChatMessage(BaseModel):
    role: str  # Possible values: 'system', 'user', 'assistant'
    content: str  # The actual message text

# Represents the structure of the data expected by the /analyze endpoint
class InputData(BaseModel):
    message: str  # New user message to be processed
    history: Optional[List[ChatMessage]] = []  # Optional previous chat history

# --- Function to detect the emotion from input text ---
def detect_emotion(text):
    """
    Runs the emotion classification model on the input text
    and returns the predicted emotion label.
    """
    try:
        # Tokenize and prepare the input text for the model
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True).to(device)
        
        # Run model inference without tracking gradients
        with torch.no_grad():
            logits = model_bert(**inputs).logits
        
        # Identify the predicted emotion class
        predicted_class = logits.argmax(dim=-1).item()
        
        # Return the corresponding emotion label
        return emotion_labels[predicted_class] if predicted_class < len(emotion_labels) else "unspecified"
    except Exception as e:
        # Log any errors during emotion detection and return fallback label
        print(f"[Emotion detection error]: {e}")
        return "unspecified"

# --- Endpoint to analyze user input and generate assistant response ---
@app.post("/analyze")
def analyze(input_data: InputData):
    """
    Accepts a message and optional history,
    detects the user's emotion, and generates a response
    using the GPT model with therapeutic guidance.
    """
    user_input = input_data.message

    # Detect emotion from the user's current message
    emotion = detect_emotion(user_input)

    # Load chat history, or initialize as empty list if not provided
    history = input_data.history or []

    # Ensure a system prompt is present to guide the assistant's behavior
    if not any(msg.role == "system" for msg in history):
        history.insert(0, ChatMessage(
            role="system",
            content=(
                "You are a compassionate therapist. Respond briefly and kindly. "
                "Always end your reply with a thoughtful question that encourages the user to keep talking."
            )
        ))

    # Append the current user message, labeled with detected emotion
    history.append(ChatMessage(role="user", content=f"User (emotion: {emotion}): {user_input}"))

    # Generate assistant response using OpenAI's GPT model
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-nano",  # Specify the OpenAI model to use
            messages=[msg.dict() for msg in history],  # Provide conversation history to maintain context
            max_tokens=128,  # Limit response length
            temperature=0.7,  # Controls randomness/creativity
            top_p=0.9,  # Controls nucleus sampling
        )
        response_text = completion.choices[0].message.content

        # Add assistant's response to the history
        history.append(ChatMessage(role="assistant", content=response_text))
    except Exception as e:
        # In case of an error, return a fallback message and log the issue
        response_text = f"[Model error]: {e}"
        history.append(ChatMessage(role="assistant", content=response_text))

    # Return emotion, response, and updated history to the client
    return {
        "emotion": emotion,
        "response": response_text,
        "updated_history": [msg.dict() for msg in history]
    }