import json
import numpy as np
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, HnswConfigDiff
from tqdm import tqdm

# --- CONFIGURATION ---
COLLECTION_NAME = "loksabha_rag_bge"
QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 1000 # Local uploads over HTTP can handle massive batches

# Define your file pairs here!
DATASETS = [
    {
        "name": "Q&As",
        "json": "patched_corpus_A.json", 
        "npy": "qa_loksabha_vectors.npy"
    },
    {
        "name": "Debates & Bills",
        "json": "final_corpus_B.json", 
        "npy": "db_loksabha_vectors.npy"
    }
]

def master_upload():
    print("Connecting to local Qdrant database...")
    client = QdrantClient(url=QDRANT_URL)
    
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating fresh collection: '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # --- THE LOW-RAM FIX ---
            # 1. Store the raw vectors on the hard drive
            vectors_config=VectorParams(
                size=1024, 
                distance=Distance.COSINE,
                on_disk=True  
            ),
            # 2. Store the massive search graph on the hard drive
            hnsw_config=HnswConfigDiff(
                on_disk=True  
            )
            # -----------------------
        )

    # Loop through both datasets automatically
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset['name']} ===")
        
        try:
            with open(dataset["json"], "r", encoding="utf-8") as f:
                records = json.load(f)
            vectors = np.load(dataset["npy"])
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {dataset['name']} - File missing: {e}")
            continue
            
        if len(records) != len(vectors):
            print(f"❌ Error in {dataset['name']}: The JSON has {len(records)} items, but the NPY has {len(vectors)} vectors!")
            continue

        print(f"Loaded {len(records)} records. Pushing to Qdrant...")
        
        for i in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Uploading {dataset['name']}"):
            batch_records = records[i : i + BATCH_SIZE]
            batch_vectors = vectors[i : i + BATCH_SIZE]
            
            points = []
            for j, doc in enumerate(batch_records):
                # Generate a stable UUID based on the chunk ID (e.g., QA_LS18_part0)
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc["id"]))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=batch_vectors[j].tolist(),
                        payload={"document_id": doc["id"], **doc["metadata"]} 
                    )
                )
                
            client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("\n✅ Master Upload Complete! Both datasets are securely in the Vector DB.")

if __name__ == "__main__":
    master_upload()