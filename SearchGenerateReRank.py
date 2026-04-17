import sqlite3
import torch
import os
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models

# ==========================================
# CONFIGURATION
# ==========================================
# PASTE YOUR API KEY HERE:
GEMINI_API_KEY = "" 

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
reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

def retrieve_context(user_question, final_top_k=10, fetch_limit=50):
    """Stage 1: Hybrid Retrieval. Stage 2: Cross-Encoder Reranking."""
    query_dense = dense_model.encode(user_question).tolist()
    query_sparse = list(sparse_model.embed([user_question]))[0]
    
    # --- STAGE 1: QDRANT HYBRID SEARCH ---
    # We fetch 25 documents instead of 5 to give the reranker more options
    search_results = q_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense_bge", limit=fetch_limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices.tolist(),
                    values=query_sparse.values.tolist()
                ),
                using="sparse_keyword",
                limit=fetch_limit
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_limit
    ).points
    
    # Pull the text for all 50 candidates from SQLite
    candidate_docs = []
    for hit in search_results:
        doc_id = hit.payload.get("document_id")
        cursor.execute("SELECT doc_type, title, raw_text FROM documents WHERE chunk_id = ?", (doc_id,))
        sql_result = cursor.fetchone()
        
        if sql_result:
            candidate_docs.append({
                "doc_type": sql_result[0],
                "title": sql_result[1],
                "text": sql_result[2],
                # Pre-format the string so it's ready for the LLM later
                "formatted": f"[SOURCE: {sql_result[0]} - {sql_result[1]}]\n{sql_result[2]}\n"
            })
            
    if not candidate_docs:
        return ""

    # --- STAGE 2: THE CROSS-ENCODER RERANKER ---
    print(f"⚖️ Reranking the top {len(candidate_docs)} chunks...")
    
    # Create pairs of [Question, Document_Text] for the model to evaluate
    pairs = [[user_question, doc["text"]] for doc in candidate_docs]
    
    # The Cross-Encoder outputs a raw logit score for every pair
    rerank_scores = reranker.predict(pairs)
    
    # Attach the new scores to our dictionary and sort them highest to lowest
    for i in range(len(candidate_docs)):
        candidate_docs[i]["rerank_score"] = rerank_scores[i]
        
    # Sort descending based on the new Cross-Encoder score
    reranked_docs = sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)
    
    # Slice off the absolute best ones to send to Gemini
    best_docs = reranked_docs[:final_top_k]
    
    # Return the combined string of the winners
    return "\n---\n".join([doc["formatted"] for doc in best_docs])


def ask_the_parliament(user_question):
    print(f"\n🔍 Searching databases for: '{user_question}'...")
    
    # 1. RETRIEVE
    context_string = retrieve_context(user_question)
    
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