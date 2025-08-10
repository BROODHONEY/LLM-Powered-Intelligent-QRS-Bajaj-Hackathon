import os
import torch
from typing import List
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class InstructorEmbedder:
    """
    Flexible embedder supporting both Instructor and SentenceTransformer models.
    - For Instructor models (names starting with "hkunlp/instructor-"), uses INSTRUCTOR.
    - Otherwise, falls back to SentenceTransformer.
    Defaults to a small Instructor model for correctness.
    """
    def __init__(self, model_name: str = "hkunlp/instructor-base"):
        device = get_device()
        self.device = device
        self.model_name = model_name
        if model_name.startswith("hkunlp/instructor-"):
            # Try to initialize INSTRUCTOR; fall back to SentenceTransformer on errors
            try:
                self.backend = "instructor"
                self.model = INSTRUCTOR(model_name, device=device)
            except Exception as e:
                print(f"[WARN] INSTRUCTOR init failed ({e}). Falling back to SentenceTransformer.")
                self.backend = "sentence-transformers"
                fallback_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.model = SentenceTransformer(fallback_name, device=device)
        else:
            self.backend = "sentence-transformers"
            self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], task: str = "Represent the document for retrieval:") -> list:
        if self.backend == "instructor":
            inputs = [[task, t] for t in texts]
            embeddings = self.model.encode(inputs, device=self.device, batch_size=32, show_progress_bar=False)
        else:
            # Special handling for E5-family: expects "query: " and "passage: " prefixes
            model_lower = (self.model_name or "").lower()
            if "e5" in model_lower:
                is_query = "question" in task.lower() or "query" in task.lower()
                prefixed = [("query: " + t) if is_query else ("passage: " + t) for t in texts]
                embeddings = self.model.encode(prefixed, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            else:
                embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return embeddings
