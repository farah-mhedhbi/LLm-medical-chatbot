# 🏥 MediBot - Medical Chatbot

A complete medical chatbot powered by LLMs, LangChain, Pinecone, and Flask.
Built using RAG (Retrieval-Augmented Generation) architecture with 100% free AI tools.

---

## 🧠 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| LLM              | Groq API (LLaMA 3.3 70b)           |
| Embeddings       | HuggingFace (all-MiniLM-L6-v2)     |
| Vector Database  | Pinecone (Free Tier)                |
| Framework        | LangChain                           |
| Backend          | Flask                               |
| Frontend         | HTML / CSS / JavaScript             |
| Dataset          | Gale Encyclopedia of Medicine       |
| Python Version   | 3.8.6                               |

---

## 📐 RAG Architecture

```
PDF (4505 pages)
      ↓
Chunks (500 chars each) → 42,950 chunks
      ↓
Vectors (384 dimensions) ──→ Pinecone Index (stored once)
                                      ↓
User Question ──→ Vector ──→ Search Pinecone ──→ Top 3 chunks
                                                       ↓
                                           Groq LLM (LLaMA 3.3 70b)
                                                       ↓
                                               Final Medical Answer
```

---

## 📁 Project Structure

```
LLM-MEDICAL-CHATBOT/
├── data/
│   └── encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf
├── research/
│   └── trials.ipynb
├── src/
│   ├── __init__.py
│   ├── helper.py        # PDF loading, chunking, embeddings
│   └── prompt.py        # LLM prompt template
├── templates/
│   └── chat.html        # Chat UI (Purple & White theme)
├── app.py               # Flask application
├── store_index.py       # Run once to index PDF in Pinecone
├── .env                 # API keys (never commit this !)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/LLM-medical-chatbot.git
cd LLM-medical-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
Create a `.env` file at the root of the project :
```
PINECONE_API_KEY="your_pinecone_key"
GROQ_API_KEY="your_groq_key"
```

Get your free API keys :
- Pinecone → https://app.pinecone.io
- Groq → https://console.groq.com

### 4. Add your PDF dataset
Place your medical PDF in the `data/` folder.

### 5. Index the PDF into Pinecone (run only once)
```bash
python store_index.py
```

### 6. Run the application
```bash
python app.py
```

Open your browser → http://127.0.0.1:8080

---

## 💬 Features

- ✅ Ask any medical question in natural language
- ✅ Answers strictly based on the Gale Encyclopedia of Medicine
- ✅ Chat history saved locally
- ✅ Professional Purple & White UI
- ✅ 100% Free (no OpenAI, no AWS required)
- ✅ Powered by LLaMA 3.3 70b via Groq

---

## 🔑 Free API Keys

| Service  | Free Tier | Link |
|----------|-----------|------|
| Pinecone | ✅ 1 free index | https://app.pinecone.io |
| Groq     | ✅ generous limits | https://console.groq.com |

---

## 📦 Requirements

```
langchain
langchain-community
langchain-groq
langchain-pinecone
langchain-huggingface
flask
sentence-transformers
pypdf
python-dotenv
pinecone
```

---

## ⚠️ Disclaimer

This chatbot is for **informational purposes only**.
It should not replace professional medical advice.
Always consult a qualified healthcare professional.

---

## 👩‍💻 Developer

**Farah MHEDHBI**
Medical Chatbot Project — 2026