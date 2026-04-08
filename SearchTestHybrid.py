import sqlite3
import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models

# ==========================================
# CONFIGURATION
# ==========================================
QDRANT_URL = "http://localhost:6333"
# Using the exact collection name from your terminal output!
COLLECTION_NAME = "loksabha_rag_hybrid_bm25_bge" 
SQLITE_DB = "loksabha_text_store.db"

print("Booting up the Hybrid Retrieval Engine...")

# 1. Connect to databases
q_client = QdrantClient(url=QDRANT_URL)
conn = sqlite3.connect(SQLITE_DB)
cursor = conn.cursor()

# 2. Load BOTH embedding models into RAM
print("Loading BGE-M3 (Dense Model)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dense_model = SentenceTransformer("BAAI/bge-m3", device=device)

print("Loading BM25 (Sparse Keyword Model)...")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def query_hybrid_database(user_question, top_k=10):
    print(f"\n🔍 Hybrid Searching for: '{user_question}'")
    
    # Step 1: Translate the question into BOTH mathematical languages
    query_dense = dense_model.encode(user_question).tolist()
    query_sparse = list(sparse_model.embed([user_question]))[0]
    
    # Step 2: The RRF Fusion Query
    try:
        search_results = q_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # Sub-Search 1: The "Vibe Check" (Semantic Meaning)
                models.Prefetch(
                    query=query_dense,
                    using="dense_bge",
                    limit=50
                ),
                # Sub-Search 2: The "Ctrl+F" (Exact Keywords)
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist()
                    ),
                    using="sparse_keyword",
                    limit=50
                )
            ],
            # Ask Qdrant to mathematically fuse the two lists
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        ).points
        
    except Exception as e:
        print(f"❌ Qdrant Search Error: {e}")
        return

    if not search_results:
        print("No matches found in Qdrant.")
        return

    # Step 3: Fetch the actual English text from SQLite
    print("-" * 60)
    for i, hit in enumerate(search_results):
        doc_id = hit.payload.get("document_id")
        # RRF scores look different from Cosine scores. They are usually smaller decimals.
        score = hit.score 
        
        # Pull the English text from your vault
        cursor.execute("SELECT raw_text, doc_type FROM documents WHERE chunk_id = ?", (doc_id,))
        sql_result = cursor.fetchone()
        
        if sql_result:
            text = sql_result[0]
            doc_type = sql_result[1]
        else:
            text = "[TEXT MISSING IN SQLITE]"
            doc_type = "UNKNOWN"
            
        preview = text[:200].replace('\n', ' ') + "..."
        
        print(f"Result {i+1} | RRF Score: {score:.4f} | Type: {doc_type} | ID: {doc_id}")
        print(f"Preview: {preview}\n")

if __name__ == "__main__":
    # Test a tricky question that requires exact phrasing AND semantic meaning
    test_question = "What are the specific penalties and fines mentioned in the Data Protection Bill?"
    query_hybrid_database(test_question)
    
    q_client.close()
    conn.close()