import sqlite3
import torch
import os
from google import genai
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models

# ==========================================
# CONFIGURATION
# ==========================================
# PASTE YOUR API KEY HERE:
GEMINI_API_KEY = "AIzaSyCIqT7DpVsh_Fk7YWUUTr82xvctQEaU8UQ" 

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "loksabha_rag_hybrid_bm25_bge" 
SQLITE_DB = "loksabha_text_store.db"

print("Booting up the Final RAG Pipeline...")


# --- REMOVE THESE LINES ---
# import google.generativeai as genai
# genai.configure(api_key=GEMINI_API_KEY)
# llm = genai.GenerativeModel('gemini-1.5-flash')

# --- REPLACE WITH THIS ---
# Initialize the new GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Connect to local databases
q_client = QdrantClient(url=QDRANT_URL)
conn = sqlite3.connect(SQLITE_DB)
cursor = conn.cursor()

# 3. Load embedding models
device = "cuda" if torch.cuda.is_available() else "cpu"
dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def retrieve_context(user_question, top_k=10):
    """Searches Qdrant and SQLite, returning a combined string of the best text."""
    query_dense = dense_model.encode(user_question).tolist()
    query_sparse = list(sparse_model.embed([user_question]))[0]
    
    search_results = q_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense_bge", limit=50),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices.tolist(),
                    values=query_sparse.values.tolist()
                ),
                using="sparse_keyword",
                limit=50
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points
    
    retrieved_texts = []
    for hit in search_results:
        doc_id = hit.payload.get("document_id")
        cursor.execute("SELECT doc_type, title, raw_text FROM documents WHERE chunk_id = ?", (doc_id,))
        sql_result = cursor.fetchone()
        
        if sql_result:
            doc_type, title, text = sql_result
            # Format the text clearly so the LLM knows where it came from
            formatted_chunk = f"[SOURCE: {doc_type} - {title}]\n{text}\n"
            retrieved_texts.append(formatted_chunk)
            
    return "\n---\n".join(retrieved_texts)

def ask_the_parliament(user_question):
    print(f"\n🔍 Searching databases for: '{user_question}'...")
    
    # 1. RETRIEVE
    context_string = retrieve_context(user_question, top_k=5)
    
    if not context_string.strip():
        print("❌ No relevant documents found in the database.")
        return

    # 2. GENERATE
    print("🧠 Reading documents and generating answer...")
    
    prompt = f"""
    You are an expert parliamentary assistant for the Government of India. 
    Your job is to answer the user's question using ONLY the provided context from Lok Sabha debates and bills.
    
    Rules:
    1. If the answer is not contained in the context, explicitly say "I cannot answer this based on the provided parliamentary records."
    2. Do not use outside knowledge. 
    3. Cite the specific Bill or Debate you are pulling the facts from in your answer.

    CONTEXT:
    {context_string}

    USER QUESTION: 
    {user_question}
    """
    
    # response = llm.generate_content(prompt)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    print("\n" + "="*60)
    print("🏛️ LOK SABHA AI ASSISTANT")
    print("="*60)
    print(response.text)
    print("="*60)

if __name__ == "__main__":
    # The ultimate test!
    question = "What are the specific penalties and fines mentioned in the Data Protection Bill?"
    ask_the_parliament(question)
    
    q_client.close()
    conn.close()