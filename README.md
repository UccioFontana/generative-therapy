
# Generative Therapy

**Generative Therapy** is a research-oriented system that combines **Virtual Reality (VR)**, **Large Language Models (LLMs)**, and **Emotion Detection** to simulate interactive therapeutic experiences. The project is part of a thesis work and explores how AI and immersive environments can be used to promote mental wellness and emotional support through natural interaction.

---

## 🧠 Project Overview

The system allows a user to interact with an AI-powered virtual therapist inside a VR game built in Unity. Each user message is analyzed for emotional content using a fine-tuned BERT model. This emotion guides both the conversational tone of the AI assistant and the immersive environment experienced by the user.

### 🔁 Workflow Summary

1. **User Input (VR Environment)**:  
   A user sends a message through a local running interface.

2. **Backend Communication (FastAPI)**:  
   The message is sent as a WebRequest to a local FastAPI server.

3. **Emotion Detection**:  
   The FastAPI backend uses a fine-tuned BERT model to detect the underlying emotion in the message.

4. **LLM Response (OpenAI)**:  
   The message (tagged with emotion) is passed to OpenAI’s GPT-4.1-nano model for an empathetic response.

5. **VR Adaptation**:  
   The detected emotion influences the virtual environment—guiding the user to an emotionally appropriate scene (e.g., calm for sadness, bright for joy).

---

## 🛠️ Technologies Used

| Component       | Description                              |
|----------------|------------------------------------------|
| FastAPI         | Backend framework for API creation       |
| Hugging Face Transformers | For loading and using pre-trained BERT models |
| PyTorch         | Deep learning inference engine           |
| OpenAI API      | GPT-4.1-nano for language generation     |
| Unity (VR)      | Frontend VR interaction environment      |
| dotenv          | Environment variable management          |

---

## 📁 Project Structure

```
generative-therapy/
│
├── main.py                  # FastAPI app with emotion detection and LLM response logic
├── .env                     # Contains OpenAI API key (not committed)
├── README.md                # Project documentation
├── data-preprocessing/      # Scripts for preparing sentiment datasets
├── fine-tuning/             # BERT fine-tuning and evaluation utilities
```

---

## 📦 Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/UccioFontana/generative-therapy.git
cd generative-therapy
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API Keys**
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
```

4. **Download the Emotion Model**
Make sure the `BERT_sent_reg` directory (fine-tuned model) is in the project root or change the path in `main.py`.

---

## 🚀 Running the Application

1. **Start FastAPI Server**
```bash
uvicorn main:app --reload
```

2. **Send a POST Request to /analyze**
Example request:
```json
{
  "message": "I feel so anxious about tomorrow.",
  "history": []
}
```

3. **Expected Response**
```json
{
  "emotion": "nervousness",
  "response": "That sounds stressful. Do you want to talk more about what’s causing your anxiety?",
  "updated_history": [...]
}
```

---

## 🔮 Future Features

- **Text-to-Speech (TTS) and Speech-to-Text (STT)** integration for voice communication.
- **More immersive Unity environments** tailored to each emotional state.
- **Multilingual support** for broader accessibility.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 👤 Author

Eustachio Fontana  
Feel free to connect via [GitHub](https://github.com/UccioFontana)

---
