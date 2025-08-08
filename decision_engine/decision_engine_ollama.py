import os
import json
import requests
from typing import List

from dotenv import load_dotenv
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def generate_answer(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Calls Ollama’s /api/generate endpoint.
    """
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }
    resp = requests.post(url, json=payload, timeout=320)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {'response': '...'} non-stream
    return data.get("response", "").strip()
