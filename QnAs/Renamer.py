import os
import json
from tqdm import tqdm

# --- CONFIGURATION ---
# Target the specific Lok Sabha folder you want to fix
TARGET_FOLDER = "QA-LS-18" 
METADATA_FILE = os.path.join(TARGET_FOLDER, "metadata.json")
PDF_DIR = os.path.join(TARGET_FOLDER, "loksabha_pdfs")

def fix_filenames():
    print(f"Loading metadata from {TARGET_FOLDER}...")
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {METADATA_FILE}")
        return

    success_count = 0
    already_fixed_count = 0
    missing_count = 0

    print(f"Checking {len(records)} records for renaming...")

    for record in tqdm(records, desc="Renaming Files", unit="file"):
        pdf_url = record.get("pdf_url")
        if not pdf_url:
            continue

        # Extract the parts from the URL
        url_parts = pdf_url.split("/")
        
        original_filename = url_parts[-1]
        if not original_filename.lower().endswith(".pdf"):
            original_filename += ".pdf"
            
        unique_id = url_parts[-3] 
        new_filename = f"{unique_id}_{original_filename}"

        # Define the full paths
        old_filepath = os.path.join(PDF_DIR, original_filename)
        new_filepath = os.path.join(PDF_DIR, new_filename)

        # 1. Check if it has ALREADY been renamed (prevents errors if you run the script twice)
        if os.path.exists(new_filepath):
            already_fixed_count += 1
            continue
            
        # 2. Check if the old file exists, then rename it
        if os.path.exists(old_filepath):
            try:
                os.rename(old_filepath, new_filepath)
                success_count += 1
            except Exception as e:
                print(f"\nError renaming {original_filename}: {e}")
        else:
            # The file is missing entirely (likely one of the ~1,500 that failed originally)
            missing_count += 1

    print("\n--- RENAMING SUMMARY ---")
    print(f"Successfully Renamed: {success_count}")
    print(f"Already Correct Format: {already_fixed_count}")
    print(f"Files Not Found: {missing_count} (Likely dead links or failed downloads)")

if __name__ == "__main__":
    fix_filenames()