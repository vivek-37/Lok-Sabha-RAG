import sqlite3
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from config import QDRANT_URL, COLLECTION_NAME, SQLITE_DB

device = "cuda" if torch.cuda.is_available() else "cpu"

dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)


q_client = QdrantClient(url=QDRANT_URL)


def retrieve_context(query, final_top_k=10, fetch_limit=50):
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    query_dense = dense_model.encode(query).tolist()
    query_sparse = list(sparse_model.embed([query]))[0]

    results = q_client.query_points(
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

    doc_ids = [r.payload.get("document_id") for r in results if r.payload.get("document_id")]

    if not doc_ids:
        conn.close()
        return ""

    placeholders = ",".join("?" * len(doc_ids))
    cursor.execute(
        f"SELECT chunk_id, doc_type, title, raw_text FROM documents WHERE chunk_id IN ({placeholders})",
        doc_ids
    )

    rows = cursor.fetchall()
    doc_map = {r[0]: r for r in rows}

    docs = []
    for doc_id in doc_ids:
        if doc_id in doc_map:
            row = doc_map[doc_id]
            docs.append({
                "text": row[3],
                "formatted": f"[SOURCE: {row[1]} - {row[2]}]\n{row[3]}"
            })

    pairs = [[query, d["text"]] for d in docs]
    scores = reranker.predict(pairs)

    for i, d in enumerate(docs):
        d["score"] = scores[i]

    docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:final_top_k]

    conn.close()

    return "\n---\n".join([d["formatted"] for d in docs])