import json

def deduplicate_internal_ids(filename, output_filename):
    print(f"Scanning {filename} for internal duplicates...")
    with open(filename, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    seen_ids = set()
    duplicate_count = 0
    
    for doc in records:
        original_id = doc["id"]
        new_id = original_id
        
        # If we've seen this ID before, append a counter until it's unique
        counter = 1
        while new_id in seen_ids:
            new_id = f"{original_id}_subchunk{counter}"
            counter += 1
            
        if new_id != original_id:
            duplicate_count += 1
            doc["id"] = new_id
            
        seen_ids.add(new_id)

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(records, f)
        
    print(f"✅ Fixed {duplicate_count} internal ID collisions!")
    print(f"Saved perfectly clean data to {output_filename}")

if __name__ == "__main__":
    deduplicate_internal_ids("patched_corpus_B.json", "final_corpus_B.json")