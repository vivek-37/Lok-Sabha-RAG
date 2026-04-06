import json

def patch_json(filename, prefix, output_filename):
    print(f"Patching {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    for doc in records:
        # If the ID is "doc_1_part1", it becomes "QA_doc_1_part1"
        if not doc["id"].startswith(prefix):
            doc["id"] = f"{prefix}_{doc['id']}"
            
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} patched records to {output_filename}\n")

# Patch your file
patch_json("chunked_rag_corpus.json", "QA", "patched_corpus_A.json")

# Patch teammate's file
patch_json("chunked_rag_corpus_B.json", "DB", "patched_corpus_B.json")