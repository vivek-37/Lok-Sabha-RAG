import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --- CONFIGURATION ---
URL_FILE = "QA-LS-16/pdf_links.txt"
DOWNLOAD_DIR = "loksabha_pdfs"

# Number of simultaneous downloads. Keep this between 5 and 10 to avoid IP bans.
MAX_WORKERS = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
}
# ---------------------

def get_robust_session():
    """Creates a requests session that automatically retries failed connections."""
    session = requests.Session()
    
    # Configure automatic retries with exponential backoff
    # If the server says "429 Too Many Requests", it will wait and try again
    retries = Retry(
        total=1,
        backoff_factor=1, # Wait times: 1s, 2s, 4s, 8s, 16s...
        status_forcelist=[429, 500, 502, 503, 504]
    )
    
    adapter = HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

def download_single_pdf(url, session):
    """Worker function to download a single file."""
    url_parts = url.split("/")
    
    original_filename = url_parts[-1]
    if not original_filename.lower().endswith(".pdf"):
        original_filename += ".pdf"
        
    unique_id = url_parts[-3]
    
    filename = f"{unique_id}_{original_filename}"
        
    save_path = os.path.join(DOWNLOAD_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(save_path):
        return True, "Already Exists"

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()

        # Check if the server secretly gave us an HTML Captcha page instead of a PDF
        if 'application/pdf' not in response.headers.get('Content-Type', ''):
             return False, "Server returned non-PDF file (Possible Captcha/Ban)"

        with open(save_path, "wb") as pdf_file:
            pdf_file.write(response.content)
            
        return True, "Success"

    except requests.exceptions.RequestException as e:
        return False, str(e)  

def fast_bulk_download():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("Reading URLs...")
    with open(URL_FILE, "r") as file:
        urls = [line.strip() for line in file if line.strip()]

    urls = urls[:(len(urls)//1)*1]
    print(f"Loaded {len(urls)} URLs. Starting concurrent download with {MAX_WORKERS} workers...")
    
    session = get_robust_session()
    success_count = 0
    fail_count = 0
    failed_urls = []

    # ThreadPoolExecutor runs our downloads in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the URLs to our worker function
        future_to_url = {executor.submit(download_single_pdf, url, session): url for url in urls}
        
        # Wrap the execution in tqdm for a sleek progress bar
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Downloading PDFs", unit="file"):
            url = future_to_url[future]
            success, message = future.result()
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_urls.append(f"{url} - Error: {message}")

    print("\n--- DOWNLOAD SUMMARY ---")
    print(f"Total Attempted: {len(urls)}")
    print(f"Successful/Skipped: {success_count}")
    print(f"Failed: {fail_count}")

    # Log the failures so you can investigate them later
    if failed_urls:
        with open("failed_downloads_log.txt", "w") as f:
            for fail in failed_urls:
                f.write(fail + "\n")
        print("Failed URLs have been written to 'failed_downloads_log.txt'")

if __name__ == "__main__":
    fast_bulk_download()