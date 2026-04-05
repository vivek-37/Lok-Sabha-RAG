import os
import json
import fitz  # PyMuPDF
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
CORPUS_FILE = "master_rag_corpus.json"
LS_FOLDERS = ["QA-LS-16","QA-LS-17", "QA-LS-18"]
INVESTIGATE_DIR = "failed_pdfs_investigation"
LOG_FILE = "failure_report.txt"

def analyze_failures():
    # 1. Gather all the IDs that successfully extracted
    print(f"Loading {CORPUS_FILE} to find successful extractions...")
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        successful_records = json.load(f)
    
    # Extract the unique_id from the end of the document string (e.g., LS18_Date_QNo_UNIQUEID)
    successful_ids = {doc["id"].split("_")[-1] for doc in successful_records}
    
    # Create a directory to drop bad files into for manual checking
    if not os.path.exists(INVESTIGATE_DIR):
        os.makedirs(INVESTIGATE_DIR)

    failed_records = []
    
    # 2. Find the missing records from the original metadata
    for folder in LS_FOLDERS:
        meta_path = os.path.join(folder, "metadata.json")
        if not os.path.exists(meta_path):
            continue
            
        with open(meta_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
            
        for record in all_metadata:
            url = record.get("pdf_url", "")
            if not url:
                continue
                
            unique_id = url.split("/")[-3]
            if unique_id not in successful_ids:
                # We found a dropped file! Let's save its data.
                record["original_folder"] = folder
                record["unique_id"] = unique_id
                failed_records.append(record)

    print(f"\nFound {len(failed_records)} total skipped files. Diagnosing causes...\n")

    reasons = {"Missing File": 0, "Corrupted/HTML (Fails to open)": 0, "Scanned Image (No Text)": 0}
    
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("--- LOK SABHA EXTRACTION FAILURE REPORT ---\n\n")
        
        for idx, record in enumerate(tqdm(failed_records, desc="Analyzing Failures")):
            url = record["pdf_url"]
            filename = url.split("/")[-1]
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"
                
            expected_filename = f"{record['unique_id']}_{filename}"
            filepath = os.path.join(record["original_folder"], "loksabha_pdfs", expected_filename)
            
            reason = ""

            # Check 1: Did it even download?
            if not os.path.exists(filepath):
                reason = "Missing File"
                reasons["Missing File"] += 1
            else:
                # Check 2: Does it open, or is it corrupted/HTML?
                try:
                    doc = fitz.open(filepath)
                    text = ""
                    for page in doc:
                        text += page.get_text("text")
                    doc.close()
                    
                    # Check 3: Does it have actual digital text?
                    if len(text.strip()) < 50:
                        reason = "Scanned Image (No Text)"
                        reasons["Scanned Image (No Text)"] += 1
                    else:
                        reason = "Unknown Error (Check manually)"
                        
                except Exception as e:
                    reason = "Corrupted/HTML (Fails to open)"
                    reasons["Corrupted/HTML (Fails to open)"] += 1

            # Log the exact URL so you can check the government website
            log.write(f"ID: {record['unique_id']} | Reason: {reason}\n")
            log.write(f"URL: {url}\n")
            log.write("-" * 50 + "\n")
            
            # Copy a sample of 20 physical files into the investigation folder so you can double check them
            if idx < 20 and os.path.exists(filepath):
                shutil.copy(filepath, os.path.join(INVESTIGATE_DIR, expected_filename))

    print("\n--- DIAGNOSTIC SUMMARY ---")
    for k, v in reasons.items():
        print(f"{k}: {v}")
    print(f"\nReport saved to {LOG_FILE}.")
    print(f"I copied 20 physical examples into the '{INVESTIGATE_DIR}' folder for you to click on and inspect.")

if __name__ == "__main__":
    analyze_failures()