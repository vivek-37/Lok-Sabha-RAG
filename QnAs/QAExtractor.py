import os
import json
import fitz  # PyMuPDF
import re
from tqdm import tqdm
from langdetect import detect, DetectorFactory

# Enforce consistent language detection results
DetectorFactory.seed = 0

# --- CONFIGURATION ---
# The folders corresponding to the different Lok Sabhas
LS_FOLDERS = ["QA-LS-16", "QA-LS-17", "QA-LS-18"]
OUTPUT_CORPUS = "master_rag_corpus.json"

def clean_parliament_text(text):
    """Strips formatting and parliamentary boilerplate."""
    boilerplate = [
        r"GOVERNMENT OF INDIA",
        r"LOK SABHA",
        r"MINISTRY OF \w+( \w+)*", 
        r"STARRED QUESTION NO\. \d+",
        r"UNSTARRED QUESTION NO\. \d+",
        r"ANSWERED ON \d{2}\.\d{2}\.\d{4}",
        r"ANSWER",
        r"\*+", 
        r"A statement is laid on the Table of the House\.",
        r"STATEMENT REFERRED TO IN REPLY TO PARTS \([a-z]\) TO \([a-z]\).*"
    ]
    
    for pattern in boilerplate:
        text = re.compile(pattern, re.IGNORECASE).sub("", text)
    
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text) 
    
    return text.strip()

def detect_text_language(text):
    """Safely detects language, defaults to 'unknown' if it fails."""
    if len(text) < 20:
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"

def build_unified_corpus():
    master_corpus = []
    total_success = 0
    total_fail = 0

    # Loop through each Lok Sabha directory
    for ls_folder in LS_FOLDERS:
        print(f"\n--- Scanning {ls_folder} ---")
        
        metadata_file = os.path.join(ls_folder, "metadata.json")
        pdf_dir = os.path.join(ls_folder, "loksabha_pdfs")

        if not os.path.exists(metadata_file):
            print(f"Error: {metadata_file} not found. Skipping...")
            continue

        with open(metadata_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        ls_number = ls_folder.split('-')[-1]

        for record in tqdm(records, desc=f"Extracting {ls_folder}", unit="doc"):
            pdf_url = record.get("pdf_url")
            
            if not pdf_url:
                total_fail += 1
                continue
                
            # ==========================================
            # NEW FILENAME LOGIC (Matching the Downloader)
            # ==========================================
            url_parts = pdf_url.split("/")
            
            original_filename = url_parts[-1]
            if not original_filename.lower().endswith(".pdf"):
                original_filename += ".pdf"
                
            # Grab the unique ID from the URL to match the files on disk
            unique_id = url_parts[-3] 
            
            # Construct the exact filename we saved during the download phase
            filename = f"{unique_id}_{original_filename}"
            # ==========================================
                
            filepath = os.path.join(pdf_dir, filename)

            if not os.path.exists(filepath):
                total_fail += 1
                continue

            try:
                # 1. Extract
                doc = fitz.open(filepath)
                full_text = ""
                for page in doc:
                    full_text += page.get_text("text") + "\n"
                doc.close()

                # 2. Clean
                cleaned_text = clean_parliament_text(full_text)

                if len(cleaned_text) < 50:
                    total_fail += 1
                    continue

                # 3. Detect Language
                lang = detect_text_language(cleaned_text)

                # Extract standard metadata points
                date = record.get("date", "UnknownDate")
                q_type = record.get("questionType", "UnknownType")
                q_no = record.get("questionNo", "UnknownNo")

                # 4. Construct JSON
                rag_document = {
                    "id": f"LS{ls_number}_{date}_{q_no}_{unique_id}", # Added unique_id here for extra safety 
                    "metadata": {
                        "lok_sabha": ls_number,
                        "date": date,
                        "title": record.get("title"),
                        "questionType": q_type,
                        "ministry": record.get("ministry"),
                        "members": record.get("members"),
                        "language": lang,
                        "source_file": filename
                    },
                    "raw_text": cleaned_text
                }
                
                master_corpus.append(rag_document)
                total_success += 1

            except Exception as e:
                total_fail += 1

    # Save the massive unified corpus
    print(f"\nExtraction complete. Saving {total_success} unified records to {OUTPUT_CORPUS}...")
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(master_corpus, f, indent=4, ensure_ascii=False)
        
    print(f"Done! {total_fail} files were skipped across all folders (missing or unreadable).")

if __name__ == "__main__":
    build_unified_corpus()