import sqlite3
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "loksabha_rag_bge"
SQLITE_DB = "loksabha_text_store.db"

print("Booting up the Retrieval Engine...")

# 1. Connect to both databases
q_client = QdrantClient(url=QDRANT_URL)
conn = sqlite3.connect(SQLITE_DB)
cursor = conn.cursor()

# 2. Load the Embedding Model (CPU is fine for single queries, it takes ~0.1 seconds)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-m3", device=device)

def query_database(user_question, top_k=3):
    print(f"\n🔍 Searching for: '{user_question}'")
    
    # Step 1: Embed the question
    query_vector = model.encode(user_question).tolist()
    
    # Step 2: Search Qdrant for the closest math
    search_results = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points
    
    if not search_results:
        print("No matches found in Qdrant.")
        return

    # Step 3: Fetch the text from SQLite
    print("-" * 50)
    for i, hit in enumerate(search_results):
        doc_id = hit.payload.get("document_id")
        score = hit.score
        
        # Pull the English text
        cursor.execute("SELECT raw_text FROM documents WHERE chunk_id = ?", (doc_id,))
        sql_result = cursor.fetchone()
        
        text = sql_result[0] if sql_result else "[TEXT MISSING IN SQLITE]"
        preview = text[:200].replace('\n', ' ') + "..."
        
        print(f"Result {i+1} | Score: {score:.4f} | ID: {doc_id}")
        print(f"Preview: {preview}\n")

if __name__ == "__main__":
    # Test your database with a real question!
    test_question = "What are the new provisions for data privacy in recent bills?"
    query_database(test_question)
    
    conn.close()