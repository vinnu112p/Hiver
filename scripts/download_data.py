"""
Data acquisition script for Customer Support on Twitter.
Supports direct automated download from mirror or Kaggle CLI.
"""
import os
import sys
import urllib.request
import time

TARGET_FILE = os.path.join("data", "raw", "twcs.csv")
DATASET_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"

def download_dataset():
    os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
    if os.path.exists(TARGET_FILE) and os.path.getsize(TARGET_FILE) > 400 * 1024 * 1024:
        print(f"[OK] Dataset already exists: {TARGET_FILE} ({os.path.getsize(TARGET_FILE)/(1024*1024):.2f} MB)")
        return TARGET_FILE

    print(f"[*] Downloading Customer Support on Twitter from mirror...")
    print(f"    Source: {DATASET_URL}")
    print(f"    Target: {TARGET_FILE}")

    req = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
    start_time = time.time()
    
    with urllib.request.urlopen(req) as response, open(TARGET_FILE, 'wb') as out_file:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024 * 4  # 4MB chunks
        last_log = time.time()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log >= 2.0 or downloaded == total_size:
                elapsed = max(0.1, now - start_time)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                percent = (downloaded / total_size * 100) if total_size else 0
                print(f"    Downloaded {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB ({percent:.1f}%) @ {speed_mb:.2f} MB/s", flush=True)
                last_log = now

    total_mb = os.path.getsize(TARGET_FILE) / (1024 * 1024)
    print(f"[OK] Download completed successfully in {time.time() - start_time:.1f}s: {total_mb:.2f} MB")
    return TARGET_FILE

if __name__ == "__main__":
    download_dataset()
