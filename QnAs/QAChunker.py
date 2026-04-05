import json
import re
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CORPUS = "master_rag_corpus.json"
OUTPUT_CORPUS = "chunked_rag_corpus.json"

# A safe limit. Most fast embedding models have a 512 token limit (~380 words).
WORD_LIMIT = 400 

def smart_split_document(text):
    """
    Splits long documents at natural structural boundaries.
    Uses positive lookahead (?=) so the trigger words aren't deleted during the split.
    Now upgraded to handle Q&As, Bills, and Committee Reports.
    """
    # Look for a newline followed immediately by any of these structural keywords
    split_pattern = r'\n(?=(?:ANNEXURE|STATEMENT REFERRED TO|APPENDIX|TABLE|CHAPTER|SECTION|RECOMMENDATION|OBSERVATION)\b)'
    
    # Split the text, ignoring case
    chunks = re.split(split_pattern, text, flags=re.IGNORECASE)
    
    # Clean up any empty chunks or pure whitespace
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]

def process_corpus():
    print(f"Loading {INPUT_CORPUS} (This might take a moment for 155k records)...")
    try:
        with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CORPUS}")
        return

    chunked_corpus = []
    split_count = 0

    print("Analyzing documents for required chunking...")
    
    for doc in tqdm(records, desc="Chunking Outliers", unit="doc"):
        raw_text = doc.get("raw_text", "")
        word_count = len(raw_text.split())
        
        # 1. If the document is short, leave it completely intact
        if word_count <= WORD_LIMIT:
            chunked_corpus.append(doc)
            continue
            
        # 2. If it's long, attempt to split it structurally
        text_chunks = smart_split_document(raw_text)
        
        # 3. Fallback: If the regex found no structural headers (a massive wall of text)
        if len(text_chunks) == 1:
            # Split by double newlines (paragraphs) to force a break
            text_chunks = re.split(r'\n{2,}', raw_text)
            text_chunks = [c for c in text_chunks if len(c.strip()) > 50]

        split_count += 1
        
        # 4. Create a unique database entry for every child chunk
        for index, chunk_text in enumerate(text_chunks):
            
            # --- CONTEXT PROPAGATION ---
            # We explicitly inject the parent's Title into the child's text
            # so the Vector DB knows exactly what this isolated chunk refers to.
            if index > 0:
                doc_type = doc.get("metadata", {}).get("type", "document")
                parent_title = doc.get("metadata", {}).get("title", "Unknown Subject")
                
                # A flexible context string that works for Q&As, Reports, and Bills
                parent_context = f"[Context: This is a continuation/section of the {doc_type} regarding '{parent_title}']\n\n"
                final_text = parent_context + chunk_text
            else:
                final_text = chunk_text

            child_doc = {
                # Append _part0, _part1 to the ID so they remain mathematically unique
                "id": f"{doc['id']}_part{index}", 
                "metadata": doc["metadata"],      # The child inherits all parent metadata
                "raw_text": final_text
            }
            chunked_corpus.append(child_doc)

    print("\n--- CHUNKING SUMMARY ---")
    print(f"Original Documents: {len(records)}")
    print(f"Long Documents Split: {split_count}")
    print(f"Total Chunks Generated: {len(chunked_corpus)}")

    print(f"\nSaving to {OUTPUT_CORPUS}...")
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(chunked_corpus, f, indent=4, ensure_ascii=False)
    
    print("Done! You are ready to split the file and start embedding.")

if __name__ == "__main__":
    process_corpus()