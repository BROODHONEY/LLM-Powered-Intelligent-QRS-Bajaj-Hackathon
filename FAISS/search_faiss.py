import numpy as np
from typing import List, Tuple
from pathlib import Path
from utils import load_pickle, DATA_DIR
from embedder import InstructorEmbedder

import faiss

from dotenv import load_dotenv
load_dotenv()

CHUNKS_PKL = DATA_DIR / "chunks.pkl"
META_PKL = DATA_DIR / "meta.pkl"
FAISS_INDEX = DATA_DIR / "faiss.index"

def load_faiss():
    if not FAISS_INDEX.exists():
        raise FileNotFoundError("FAISS index not found. Build it first.")
    index = faiss.read_index(str(FAISS_INDEX))
    chunks = load_pickle(CHUNKS_PKL)
    metas = load_pickle(META_PKL)
    return index, chunks, metas

def search(query: str, embedder: InstructorEmbedder, k: int = 5) -> List[Tuple[int, float]]:
    index, chunks, _ = load_faiss()
    qvec = np.array(embedder.encode([query], task="Represent the question for retrieving supporting documents:"), dtype="float32")
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    # Returns list of (idx, score)
    return list(zip(I[0].tolist(), D[0].tolist()))
