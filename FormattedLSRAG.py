import sqlite3
import torch
import os
import json
import streamlit as st
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models

# ==========================================
# CONFIGURATION
# ==========================================
GEMINI_API_KEY = "AIzaSyCIqT7DpVsh_Fk7YWUUTr82xvctQEaU8UQ" # Replace with your key
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

client, q_client, dense_model, sparse_model, reranker = load_systems()

# ==========================================
# 1. RAG PIPELINE (RETRIEVAL)
# ==========================================
def retrieve_context(user_question, final_top_k=10, fetch_limit=50):
    """Stage 1: Hybrid Retrieval. Stage 2: Cross-Encoder Reranking."""

    client, q_client, dense_model, sparse_model, reranker = load_systems()

    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    query_dense = dense_model.encode(user_question).tolist()
    query_sparse = list(sparse_model.embed([user_question]))[0]
    
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
    
    doc_ids = [hit.payload.get("document_id") for hit in search_results if hit.payload.get("document_id")]
    
    if not doc_ids:
        conn.close()
        return ""

    placeholders = ','.join('?' * len(doc_ids))
    query = f"SELECT chunk_id, doc_type, title, raw_text FROM documents WHERE chunk_id IN ({placeholders})"
    cursor.execute(query, doc_ids)
    sql_results = cursor.fetchall()
    
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

    pairs = [[user_question, doc["text"]] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False)
    
    for i in range(len(candidate_docs)):
        candidate_docs[i]["rerank_score"] = rerank_scores[i]
        
    reranked_docs = sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)
    best_docs = reranked_docs[:final_top_k]
    
    conn.close()
    return "\n---\n".join([doc["formatted"] for doc in best_docs])

# ==========================================
# 2. FORMATTER AGENT
# ==========================================
def formatter_agent(raw_llm_answer):
    """
    Takes the raw, dense legal output and structures it into JSON
    containing a 'simple' view and a 'detailed' view.
    """
    formatter_prompt = f"""
    You are a Formatting Agent. Your job is to take the following dense parliamentary answer 
    and convert it into two distinct representations.
    
    1. "simple": A highly accessible, 8th-grade reading level summary. Strip away heavy legal jargon, 
       keep the core facts, and use bullet points if helpful. Explain it simply.
    2. "detailed": A beautifully formatted, structured markdown version of the original text. 
       Keep all citations, clauses, and nuances intact. Enhance readability with bolding and headers.
       
    Output ONLY a valid JSON object with the keys "simple" and "detailed".
    
    RAW ANSWER TO FORMAT:
    {raw_llm_answer}
    """
    
    # Force the model to return strict JSON using the config
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=formatter_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1 # Keep it focused, no creative hallucination
        )
    )
    
    # Parse the string into a Python dictionary
    return json.loads(response.text)

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Lok Sabha AI", page_icon="🏛️", layout="centered")

st.title("🏛️ Lok Sabha AI Assistant")
st.markdown("Ask questions about Indian parliamentary debates, bills, and legislative records.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages using Tabs for the Assistant
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            # Create the UI Toggle mechanism natively
            tab_simple, tab_detailed = st.tabs(["📝 Simple View", "🏛️ Detailed Record"])
            with tab_simple:
                st.markdown(message["simple"])
            with tab_detailed:
                st.markdown(message["detailed"])

# Accept user input
if prompt := st.chat_input("E.g., Which Bills have more than 2 amendments?"):
    
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process Assistant Response
    with st.chat_message("assistant"):
        with st.status("Analyzing Parliamentary Records...", expanded=True) as status:
            
            # --- STEP A: RETRIEVE ---
            st.write("🔍 Running Hybrid Retrieval & Reranking...")
            context_string = retrieve_context(prompt)
            
            if not context_string.strip():
                status.update(label="No records found.", state="error")
                st.error("❌ No relevant documents found in the database.")
                st.stop()
                
            # --- STEP B: GENERATE RAW ANSWER ---
            st.write("🧠 Synthesizing legal facts...")
            st.write("Reranking top results with BGE-M3...")
            status.update(label="Reading documents & generating answer...", state="running")

            base_prompt = f"""You are a precision-focused legal and parliamentary assistant for the Government of India.
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
            try:
                raw_response = client.models.generate_content(
                    model='gemini-3-flash-preview',
                    contents=base_prompt
                ).text
                
                # --- STEP C: FORMATTER AGENT ---
                st.write("✍️ Formatting into Simple and Detailed views...")
                formatted_data = formatter_agent(raw_response)
                
                status.update(label="Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Error generating response.", state="error")
                st.error(f"API Error: {e}")
            
        # Render the interactive tabs for the current response
        tab_simple, tab_detailed = st.tabs(["📝 Simple View", "🏛️ Detailed Record"])
        
        with tab_simple:
            st.markdown(formatted_data["simple"])
            
        with tab_detailed:
            st.markdown(formatted_data["detailed"])
            with st.expander("View Retrieved Context (Database Hits)"):
                st.text(context_string)
                
        # Save both views to history so they persist if the user asks another question
        st.session_state.messages.append({
            "role": "assistant", 
            "simple": formatted_data["simple"], 
            "detailed": formatted_data["detailed"]
        })