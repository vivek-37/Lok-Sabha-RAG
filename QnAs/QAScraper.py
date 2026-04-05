import requests
import time
import json
import os

# Create a folder to save our data
os.makedirs("loksabha_qa_data", exist_ok=True)

# API Configuration
BASE_URL = "http://eparlib.sansad.in/restv3/fetch/all"
TOTAL_RECORDS = 80000
CHUNK_SIZE = 10000 # Adjust this based on what the server handles (try 500 or 1000 later)

all_pdf_links = []
metadata_db = []

print("Starting extraction...")

# Loop through the API using the 'start' parameter
for start_index in range(0, TOTAL_RECORDS, CHUNK_SIZE):
    print(f"Fetching records {start_index} to {start_index + CHUNK_SIZE}...")
    
    # Strictly use http:// (not https://)
    exact_url = f"http://eparlib.sansad.in/restv3/fetch/all?start={start_index}&rows={CHUNK_SIZE}&order=date_asc&collectionId=3&loksabhaNo=(16)&locale=en"
    
    # Add a standard User-Agent header to pretend to be a web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        # It looks like this API uses GET requests with query parameters
        response = requests.get(exact_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        records = data.get('records', [])
        
        for record in records:
            # 1. Get the PDF link
            files = record.get('files', [])
            if files and len(files) > 0:
                pdf_url = files[0]
                all_pdf_links.append(pdf_url)
                
                # 2. Extract useful metadata for the RAG database
                clean_record = {
                    "pdf_url": pdf_url,
                    "date": record.get("date"),
                    "title": record.get("title"),
                    "questionNo": record.get("questionNo"),
                    "questionType": record.get("questionType"),
                    "ministry": record.get("ministry", [])[0] if record.get("ministry") else None,
                    "members": record.get("members", [])
                }
                metadata_db.append(clean_record)
                
        # Be polite to the server
        time.sleep(5)
        
    except Exception as e:
        print(f"Error fetching at start index {start_index}: {e}")

# Save the links for downloading
with open("loksabha_qa_data/pdf_links.txt", "w") as f:
    for link in all_pdf_links:
        f.write(link + "\n")

# Save the metadata for your vector database
with open("loksabha_qa_data/metadata.json", "w") as f:
    json.dump(metadata_db, f, indent=4)

print(f"\nSuccess! Extracted {len(all_pdf_links)} PDF links and metadata records.")