import os
import torch
from typing import List
from InstructorEmbedding import INSTRUCTOR

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class InstructorEmbedder:
    """
    Uses Instructor models; choose a smaller model for speed if desired:
    - sentence-transformers/all-MiniLM-L6-v2
    - hkunlp/instructor-xl (heavier)
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        device = get_device()
        self.device = device
        self.model = INSTRUCTOR(model_name, device=device)
        # INSTRUCTOR internally uses device from torch; ensure it’s set globally
        torch_device = torch.device(device)
        # No explicit .to(device) needed for InstructorEmbedding wrapper.

    def encode(self, texts: List[str], task: str = "Represent the document for retrieval:") -> list:
        # Instructor requires pairs [instruction, text]
        inputs = [[task, t] for t in texts]
        embeddings = self.model.encode(inputs, device=self.device, batch_size=32, show_progress_bar=False)
        return embeddings
