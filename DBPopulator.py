import sqlite3
import json
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# Update this variable to point to the JSON file you are processing right now.
# E.g., "part1_corpus.json" or "teammate_chunked_corpus.json"
CORPUS_FILE = "final_corpus_B.json"
SQLITE_DB = "loksabha_text_store.db"
# ==========================================

def build_text_database():
    print(f"Connecting to SQLite database: {SQLITE_DB}...")
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    # Create the flexible schema.
    # 'chunk_id' is the PRIMARY KEY. This is the magic that prevents duplicates
    # and allows you to seamlessly append your teammate's data.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            chunk_id TEXT PRIMARY KEY,
            doc_type TEXT,
            title TEXT,
            raw_text TEXT
        )
    ''')
    conn.commit()

    print(f"Loading {CORPUS_FILE}...")
    try:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {CORPUS_FILE} in this directory.")
        return

    print(f"Inserting {len(records)} records into SQLite...")
    
    # Track statistics for your peace of mind
    inserted_count = 0
    ignored_count = 0

    for doc in tqdm(records, desc="Saving Text Vault"):
        chunk_id = doc["id"]
        raw_text = doc["raw_text"]
        
        # Safely extract metadata, defaulting to generic tags if missing
        meta = doc.get("metadata", {})
        doc_type = meta.get("type", "UNKNOWN").upper()
        title = meta.get("title", "Unknown Title")

        try:
            # The INSERT OR IGNORE command perfectly handles the merge.
            # If the chunk_id already exists, SQLite silently skips it.
            cursor.execute('''
                INSERT OR IGNORE INTO documents (chunk_id, doc_type, title, raw_text)
                VALUES (?, ?, ?, ?)
            ''', (chunk_id, doc_type, title, raw_text))
            
            # Check if the row was actually inserted or skipped
            if cursor.rowcount == 1:
                inserted_count += 1
            else:
                ignored_count += 1
                
        except Exception as e:
            print(f"\n⚠️ Database error on chunk {chunk_id}: {e}")

    # Lock in the changes and close the connection
    conn.commit()
    conn.close()

    print("\n✅ Text Database Sync Complete!")
    print(f"-> Successfully inserted: {inserted_count}")
    print(f"-> Skipped (already existed): {ignored_count}")
    print("-> You are safe to process the next file.")

if __name__ == "__main__":
    build_text_database()