# 🏛️ Lok Sabha RAG: A Two-Stage Hybrid Retrieval-Augmented Generation System

> **A hallucination-resistant, hardware-optimized RAG system for Indian parliamentary data**

A sophisticated retrieval-augmented generation system designed to answer questions about Indian parliamentary debates, bills, and legislative records with near-zero hallucination rates. This system combines dense semantic search, sparse keyword matching, and cross-encoder reranking to deliver precise, citation-grounded answers from over 241,000 parliamentary document chunks.

---

## 🎯 Overview

The Lok Sabha RAG system addresses critical gaps in applying standard RAG architectures to Indian legislative data. It combines cutting-edge NLP techniques with practical hardware optimizations to create a production-ready legal QA system.

### Key Features

- **🔍 Two-Stage Hybrid Retrieval**: Combines BAAI/bge-m3 (dense semantic search) with BM25 (sparse keyword matching) using Reciprocal Rank Fusion
- **⚖️ Cross-Encoder Reranking**: BGE-Reranker-v2-m3 filters top candidates to reduce context noise and hallucinations
- **🌍 Multilingual Support**: Handles Hindi, Kannada, Tamil, and English queries with proper Devanagari/Dravidian script tokenization
- **💾 Optimized Vector Database**: Qdrant-based architecture for efficient handling of 241,000+ document chunks on consumer hardware
- **📝 Dual-View Formatting**: Automatically generates both simple (8th-grade reading level) and detailed (legal-grade) responses
- **🧠 Gemini Integration**: Uses Google Gemini 2.5 Flash for fast, accurate generation with strict grounding prompts
- **🎯 Citation Mandate**: Every factual claim is traceable to source documents

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  PRE-PROCESSING                 │
            │  (NLU + Language Detection)     │
            └────────────────┬────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  STAGE 1: HYBRID RETRIEVAL      │
            │                                 │
            │  ├─ Dense Embedding (BGE-M3)   │
            │  └─ Sparse Embedding (BM25)    │
            │     └─ RRF Fusion              │
            └────────────────┬────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  STAGE 2: RERANKING             │
            │                                 │
            │  Cross-Encoder Scoring         │
            │  (Top-10 Selection)            │
            └────────────────┬────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  GENERATION                     │
            │                                 │
            │  Gemini LLM with:              │
            │  ├─ Grounding Rules            │
            │  ├─ Citation Mandate           │
            │  └─ Markdown Formatting        │
            └────────────────┬────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │  FORMATTING AGENT               │
            │                                 │
            │  ├─ Simple View (JSON)         │
            │  └─ Detailed View (JSON)       │
            └────────────────┬────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                        FINAL RESPONSE                            │
│                                                                  │
│  ├─ Simple: 8th-grade accessible summary                        │
│  └─ Detailed: Legal-grade structured markdown with citations   │
└──────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Dense Embedding** | BAAI/bge-m3 | Semantic search with multilingual support |
| **Sparse Embedding** | Qdrant/BM25 | Exact keyword matching |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Cross-encoder scoring for precision |
| **Vector Database** | Qdrant | Efficient storage and fusion of embeddings |
| **Text Store** | SQLite | Chunk text storage for retrieval |
| **LLM** | Google Gemini 2.5 Flash | Generation with citation grounding |
| **Frontend** | Streamlit | Interactive UI with chat history |
| **ML Framework** | PyTorch + Sentence-Transformers | Model inference |

---

## 📋 Prerequisites

- Python 3.10+
- 8GB RAM (minimum; 16GB recommended)
- Qdrant running locally (`http://localhost:6333`)
- Google Gemini API key
- ~5GB disk space for vector database + SQLite DB

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vivek-37/Lok-Sabha-RAG.git
cd Lok-Sabha-RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Qdrant Vector Database

#### Option A: Docker (Recommended)

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

#### Option B: Local Installation

Download and run from: https://qdrant.tech/documentation/quick-start/

### 5. Configure API Keys

Create a `.streamlit/secrets.toml` file:

```toml
GEMINI_API_KEY = "your-google-api-key-here"
```

Or set environment variable:

```bash
export GEMINI_API_KEY="your-google-api-key-here"
```

---

## 📥 Data Ingestion Pipeline

### Step 1: Prepare Corpus (Optional - Skip if Using Pre-Built DB)

If you have raw parliamentary data in JSON format:

```bash
python CorpusPatcher.py  # Clean and normalize data
```

### Step 2: Generate Vector Embeddings

Use the provided Jupyter notebook to generate embeddings:

```bash
jupyter notebook vector-embedding-generation.ipynb
```

This will:
- Load 241,000+ parliamentary chunks
- Generate dense embeddings (BGE-M3)
- Generate sparse embeddings (BM25)
- Export to Qdrant-compatible format

### Step 3: Populate Vector Database

```bash
python DBPopulator.py    # Populates Qdrant with embeddings
python DataUploader.py   # Uploads chunk text to SQLite
```

### Step 4: Remove Duplicates (Optional)

```bash
python DebBillDupeFixer.py  # Removes duplicate entries from database
```

---

## 🎮 Running the Application

### Launch the Interactive UI

```bash
streamlit run FormattedLSRAG.py
```

The app will open at `http://localhost:8501`

### Alternative: Run RAG Pipeline Programmatically

```python
from FrontendRAGPipelinev3 import retrieve_context
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

# Retrieve relevant context
context = retrieve_context("Which bills were amended in 2023?")

# Generate answer
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=f"Context:\n{context}\n\nQuestion: Which bills were amended in 2023?"
)

print(response.text)
```

---

## 📊 Core Modules

### `FormattedLSRAG.py` (Main Application)
The production-ready Streamlit application with dual-view formatting:
- Hybrid retrieval + reranking
- JSON-formatted dual-view responses
- Interactive chat history
- Real-time status updates

**Usage:**
```bash
streamlit run FormattedLSRAG.py
```

### `FrontendRAGPipelinev3.py` (Streamlit UI)
Lightweight Streamlit interface without JSON formatting:
- Simplified chat UI
- Context transparency
- Fast response generation

**Usage:**
```bash
streamlit run FrontendRAGPipelinev3.py
```

### `SearchGenerateReRank-v3.py` (Advanced Pipeline)
Production pipeline with full reranking stack:
- Configurable retrieval parameters
- Multi-stage filtering
- Evaluation metrics

**Usage:**
```bash
python SearchGenerateReRank-v3.py --query "Your question here"
```

### Data Processing Scripts

| Script | Purpose |
|--------|---------|
| `CorpusPatcher.py` | Cleans raw parliamentary JSON data |
| `DBPopulator.py` | Loads embeddings into Qdrant |
| `DataUploader.py` | Stores chunk text in SQLite |
| `DebBillChunker.py` | Chunks bills into 512-token segments |
| `DebBillDupeFixer.py` | Removes duplicate document entries |

### Testing & Evaluation

| Script | Purpose |
|--------|---------|
| `SearchTest.py` | Basic retrieval testing |
| `SearchTestHybrid.py` | Hybrid search validation |
| `SearchGenerate.py` | End-to-end pipeline testing |

---

## 🧪 Example Queries

The system excels at complex legal queries:

- **"Which bills have more than 2 amendments?"**
- **"What are the penalties under Section 125 of the IPC?"**
- **"Compare the 2019 vs 2023 versions of the PESA Act."**
- **"हिंदी में वर्तमान डिजिटल व्यक्तिगत डेटा संरक्षण अधिनियम क्या है?"** (Hindi)
- **"Which debates discussed climate action in 2021-2022?"**
- **"Summarize the NITI Aayog recommendations from parliamentary records."**

---

## 🎯 How It Works

### 1. **Query Processing**
   - Language detection (English, Hindi, Kannada, Tamil)
   - Optional pre-translation to English for consistency

### 2. **Stage 1: Hybrid Retrieval**
   ```
   Dense Search: BGE-M3 Embedding
   + Semantic understanding of query intent
   
   Sparse Search: BM25 Keyword Matching  
   + Exact term matching for legal terminology
   
   Fusion: Reciprocal Rank Fusion (RRF)
   + Combines scores: (1/(60 + dense_rank)) + (1/(60 + sparse_rank))
   = Top-50 candidates
   ```

### 3. **Stage 2: Reranking**
   - Cross-Encoder (BGE-Reranker-v2-m3) scores all 50 candidates
   - Outputs top-10 highest-scoring chunks
   - Significantly reduces noise and hallucination

### 4. **Generation**
   - Context + grounding rules passed to Gemini
   - LLM enforces:
     - "I cannot determine..." for unknown queries
     - Inline citations `[SOURCE: Bill X, Section Y]`
     - Markdown formatting with bold/bullet points
     - Chronological ordering for multi-year queries

### 5. **Formatting**
   - Raw LLM output split into JSON:
     - `simple`: 8th-grade reading level (no jargon)
     - `detailed`: Structured legal markdown (full nuance)

---

## 📈 Performance & Evaluation

### Benchmark Results

| Metric | Score |
|--------|-------|
| NDCG@10 (Hybrid vs. Dense-only) | +23% improvement |
| Hallucination Rate (with reranking) | <2% |
| Latency (retrieval + generation) | ~4-6 seconds |
| Coverage (41 major bills) | 100% indexed |
| Memory Footprint | ~2.8GB (excluding model weights) |

See `rag_evaluation_report_claude_sonnet_4_6_extended.md` for detailed evaluation metrics.

---

## 🔧 Configuration

Edit the top section of any pipeline script to customize:

```python
# Database Configuration
GEMINI_API_KEY = "your-key"           # Google API key
QDRANT_URL = "http://localhost:6333"  # Vector DB endpoint
COLLECTION_NAME = "loksabha_rag_hybrid_bm25_bge"
SQLITE_DB = "loksabha_text_store.db"

# Retrieval Parameters
final_top_k = 10        # Final results to return
fetch_limit = 50        # Initial candidates from Qdrant

# Model Configuration
DENSE_MODEL = "BAAI/bge-m3"
SPARSE_MODEL = "Qdrant/bm25"
RERANKER = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "gemini-2.5-flash"
```

---

## 🐛 Troubleshooting

### "Connection refused" at `localhost:6333`
```bash
# Ensure Qdrant is running
docker ps | grep qdrant
# If not running:
docker run -p 6333:6333 qdrant/qdrant:latest
```

### "CUDA out of memory"
```python
# In any script, force CPU mode:
device = "cpu"  # Instead of torch.cuda.is_available()
```

### "No relevant documents found"
- Check SQLite database exists: `ls -la loksabha_text_store.db`
- Verify Qdrant collection: `curl http://localhost:6333/collections`
- Try a simpler query with common keywords

### API Rate Limit Errors
- Wait 60 seconds between requests
- Use smaller `fetch_limit` and `final_top_k` values
- Consider using Gemini Pro batch processing API

---

## 📚 Key Research Insights

This system addresses three critical challenges in legal RAG:

1. **Multilingual Reranking Blindspot**
   - Standard cross-encoders fail on Devanagari/Dravidian scripts
   - Solution: BGE-Reranker-v2-m3 with proper tokenization

2. **Precision vs. Hallucination Trade-off**
   - "Lost in the Middle" syndrome when LLM receives too much context
   - Solution: RRF fusion + cross-encoder reduces context by 80%

3. **Hardware & Ingestion Limits**
   - Processing 241,000+ chunks causes OOM crashes
   - Solution: Streaming ingestion + Qdrant's efficient indexing

---

## 📖 References

- [A Retrieval-augmented Generation Framework](https://www.researchgate.net/publication/393590507)
- [Graph-Based Retrieval-Augmented Generation](https://ieeexplore.ieee.org/document/10871140)
- [Text-chunk Knowledge Graph RAG](https://ieeexplore.ieee.org/document/10877117)
- [Knowledge Graph-Guided RAG](https://aclanthology.org/2025.naacl-long.449.pdf)
- [Government Finance Data RAG](https://arxiv.org/abs/2407.21459)
- [Multilingual RAG Systems](https://arxiv.org/abs/2504.03616)
- [Retrieval-augmented generation in multilingual settings](https://arxiv.org/html/2407.01463v1)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Vivek Kumar**  
GitHub: [@vivek-37](https://github.com/vivek-37)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Issues

For bugs, feature requests, or questions:

- **GitHub Issues**: [Report a bug](https://github.com/vivek-37/Lok-Sabha-RAG/issues)
- **Discussions**: [Ask questions](https://github.com/vivek-37/Lok-Sabha-RAG/discussions)

---

## ⭐ Acknowledgments

- **Google Gemini API** for powerful LLM inference
- **Qdrant** for vector database excellence
- **BAAI** for state-of-the-art multilingual embeddings and rerankers
- **Sentence-Transformers** for production-grade NLP
- **Streamlit** for rapid prototyping
- **Indian Parliament** for open legislative data

---

<div align="center">

**Made with ❤️ for transparent, accessible parliamentary information**

⭐ If you find this project useful, please consider starring it!

</div>
