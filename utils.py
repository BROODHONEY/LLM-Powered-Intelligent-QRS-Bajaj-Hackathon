import os
import re
import io
import json
import time
import math
import pdfplumber
import requests
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = Path("vector_store")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def timer_ms() -> int:
    return int(time.time() * 1000)

def read_url_text(url: str) -> str:
    # Basic content fetcher for PDFs or text/HTML; extend as needed
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            text = "\n".join([p.extract_text() or "" for pg in pdf.pages for p in [pg]])
        return text
    else:
        # crude text extraction; you can add readability or BeautifulSoup
        text = resp.text
        # strip scripts/styles crudely
        text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.S)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks

def save_pickle(obj, path: Path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
