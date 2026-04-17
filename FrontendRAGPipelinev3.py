import sqlite3
import torch
import os
import json
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models

# ==========================================
# CONFIGURATION
# ==========================================
GEMINI_API_KEY = "" # Replace or use st.secrets
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "loksabha_rag_hybrid_bm25_bge" 
SQLITE_DB = "loksabha_text_store.db"

# ==========================================
# CACHED INITIALIZATION
# ==========================================
@st.cache_resource
def load_systems():
    """Loads models and DB connections only once to prevent memory crashes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    q_client = QdrantClient(url=QDRANT_URL)
    
    dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)
    
    return client, q_client, dense_model, sparse_model, reranker

# Load everything into memory
client, q_client, dense_model, sparse_model, reranker = load_systems()

# ==========================================
# PIPELINE FUNCTIONS (Adapted for UI)
# ==========================================
def retrieve_context(user_question, final_top_k=10, fetch_limit=50):
    """Stage 1: Hybrid Retrieval. Stage 2: Cross-Encoder Reranking."""
    
    # --- CHANGE 1: GRAB CACHED MODELS ---
    # By calling your @st.cache_resource function here, you guarantee 
    # you are using the models already loaded in memory.
    client, q_client, dense_model, sparse_model, reranker = load_systems()

    # --- CHANGE 2: THREAD-SAFE SQLITE ---
    # Open the connection inside the function so Streamlit doesn't throw a thread error
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    # Create your embeddings using the cached models
    query_dense = dense_model.encode(user_question).tolist()
    query_sparse = list(sparse_model.embed([user_question]))[0]
    
    # --- STAGE 1: QDRANT HYBRID SEARCH ---
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
    
    # Extract all document IDs from the Qdrant hits
    doc_ids = [hit.payload.get("document_id") for hit in search_results if hit.payload.get("document_id")]
    
    if not doc_ids:
        conn.close() # Always close before returning
        return ""

    # --- OPTIMIZATION 2: BULK SQLITE FETCH ---
    placeholders = ','.join('?' * len(doc_ids))
    query = f"SELECT chunk_id, doc_type, title, raw_text FROM documents WHERE chunk_id IN ({placeholders})"
    cursor.execute(query, doc_ids)
    sql_results = cursor.fetchall()
    
    # Create a dictionary for O(1) lookups
    doc_map = {
        row[0]: {"doc_type": row[1], "title": row[2], "text": row[3]} 
        for row in sql_results
    }

    candidate_docs = []
    for doc_id in doc_ids:
        if doc_id in doc_map:
            doc_data = doc_map[doc_id]
            candidate_docs.append({
                "text": doc_data["text"],
                "formatted": f"[SOURCE: {doc_data['doc_type']} - {doc_data['title']}]\n{doc_data['text']}\n"
            })

    # --- STAGE 2: THE CROSS-ENCODER RERANKER ---
    print(f"⚖️ Reranking the top {len(candidate_docs)} chunks...")
    
    pairs = [[user_question, doc["text"]] for doc in candidate_docs]
    
    # The reranker will run fast because it is already cached in memory
    rerank_scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False)
    
    for i in range(len(candidate_docs)):
        candidate_docs[i]["rerank_score"] = rerank_scores[i]
        
    reranked_docs = sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)
    best_docs = reranked_docs[:final_top_k]
    
    # Safely close the database connection
    conn.close()
    
    return "\n---\n".join([doc["formatted"] for doc in best_docs])

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Lok Sabha RAG", page_icon="🏛️", layout="centered")

st.title("🏛️ Lok Sabha RAG")
st.markdown("Ask questions about Indian parliamentary debates, bills, and legislative records.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your query here."):
    
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.status("Searching Parliamentary Records...", expanded=True) as status:
            st.write("Running Hybrid Retrieval (Dense + BM25)...")
            # # Connect to SQLite per thread
            # conn = sqlite3.connect(SQLITE_DB)
            # cursor = conn.cursor()
            
            # Execute retrieval
            context_string = retrieve_context(prompt) # Ensure your function returns the context
            
            if not context_string.strip():
                status.update(label="No records found.", state="error")
                st.error("❌ No relevant documents found in the database.")
                st.stop()
                
            st.write("Reranking top results with BGE-M3...")
            status.update(label="Reading documents & generating answer...", state="running")
            
            # Formulate prompt
            system_prompt = f"""You are a precision-focused legal and parliamentary assistant for the Government of India.
            Your objective is to synthesize accurate, highly structured answers to the user's query using ONLY the provided Lok Sabha records.

            CORE DIRECTIVES:
            1. ABSOLUTE GROUNDING: You must rely exclusively on the CONTEXT provided below. Do not use external knowledge, internet training data, or infer legal facts not explicitly stated.
            2. THE "PIVOT & INFORM" RULE: If the CONTEXT does not contain the exact information required to fully answer the query, you MUST begin your response by explicitly stating: "I cannot directly answer this specific question based on the provided parliamentary records." Immediately following this, you must pivot to provide a structured summary of the most closely related legal facts, debates, or definitions that *are* present in the retrieved CONTEXT.
            3. CITATION MANDATE: Every factual claim, statistic, or penalty in your answer (including the related information provided under Rule 2) MUST be immediately followed by an inline citation to its source document (e.g., [SOURCE: BILL - The Personal Data Protection Bill, 2019]). 
            4. CHRONOLOGY & CONFLICTS: If the CONTEXT contains multiple documents showing how a law or debate evolved over time (e.g., a 2019 draft vs. a 2023 amendment), clearly separate them and explain the evolution. 
            5. STRUCTURAL CLARITY: Output your response using strict Markdown. Use bullet points for lists, and use bold text for specific penalties, dates, and Section/Clause numbers. Maintain a formal, objective, and non-partisan tone.

            CONTEXT:
            {context_string}

            USER QUESTION: 
            {prompt}

            FINAL ANSWER:
            """
            
            # Call Gemini
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash', # Or whichever you prefer
                    contents=system_prompt
                )
                answer = response.text
                status.update(label="Complete!", state="complete", expanded=False)
                
                # Render the final answer
                st.markdown(answer)
                
                # Optional: Show the context used in an expander
                with st.expander("View Retrieved Context"):
                    st.text(context_string)
                    
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                status.update(label="Error generating response.", state="error")
                st.error(f"API Error: {e}")
            
            # finally:
            #     conn.close()