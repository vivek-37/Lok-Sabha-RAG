# 🏛️ Lok Sabha RAG System

A **Retrieval-Augmented Generation (RAG)** system for querying Indian parliamentary data (Lok Sabha debates, bills, and records) using hybrid search and LLM-based answer synthesis.

---

## 🚀 Features

* 🔍 **Hybrid Retrieval** (Dense + BM25 via Qdrant)
* ⚖️ **Cross-Encoder Reranking** (BGE reranker)
* 🧠 **LLM Answer Generation** using Gemini
* 🎨 **Structured Output Formatting** using OpenAI
* 💬 **Interactive Chat UI** with Streamlit
* 🔐 Secure configuration via `.env`

---

## 📁 Project Structure

```
loksabha_rag/
│
├── app.py              # Streamlit UI + main pipeline
├── rag_pipeline.py     # Retrieval + reranking logic
├── llm.py              # Gemini + OpenAI formatting layer
├── config.py           # Environment variable loader
│
├── .env                # API keys & configs (not committed)
├── .gitignore          # Ignore sensitive/unnecessary files
├── requirements.txt    # Dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd loksabha_rag
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=loksabha_rag_hybrid_bm25_bge
SQLITE_DB=loksabha_text_store.db
```

---

## 🧠 System Workflow

```
User Query
   ↓
Hybrid Retrieval (Qdrant: Dense + Sparse)
   ↓
SQLite Document Fetch
   ↓
Cross-Encoder Reranking
   ↓
Gemini → Structured JSON Answer
   ↓
OpenAI → Clean Markdown Formatting
   ↓
Streamlit UI Display
```

---

## ▶️ Running the Application

Make sure:

* Qdrant is running locally (`localhost:6333`)
* SQLite DB file exists

Then run:

```bash
streamlit run app.py
```

App will open in your browser.

---

## 📌 Notes

* First run may take time due to model loading.
* GPU is recommended for faster reranking but not required.
* Ensure your Qdrant collection and SQLite DB are properly populated.

---

## 🔮 Future Improvements

* Replace OpenAI formatter with direct JSON → UI rendering (faster)
* Add streaming responses
* Add evaluation metrics (RAGAS)
* Deploy using FastAPI + Streamlit frontend

---

## 🛡️ Security

* Never commit `.env`
* Rotate API keys periodically
* Use environment-based configs in production

---

## 👨‍💻 Authors

* Vivek Rekha Ashoka
* Vrishant Bhalla

---

## 📜 License

This project is for academic/research purposes.