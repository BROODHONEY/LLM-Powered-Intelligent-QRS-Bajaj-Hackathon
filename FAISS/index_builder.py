import os
import numpy as np
from pathlib import Path
from typing import List, Dict
from utils import DATA_DIR, save_pickle
from embedder import InstructorEmbedder

try:
    import faiss
except ImportError:
    raise RuntimeError("FAISS not installed. Use faiss-gpu or faiss-cpu per environment.")

from dotenv import load_dotenv
load_dotenv()

CHUNKS_PKL = DATA_DIR / "chunks.pkl"
FAISS_INDEX = DATA_DIR / "faiss.index"
META_PKL = DATA_DIR / "meta.pkl"

def build_index(chunks: List[str], metas: List[Dict], embedder: InstructorEmbedder, use_ivf: bool = False, nlist: int = 256):
    vectors = np.array(embedder.encode(chunks), dtype="float32")
    d = vectors.shape[1]
    if use_ivf:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(vectors)
        index.train(vectors)
        index.add(vectors)
    else:
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(vectors)
        index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX))
    save_pickle(chunks, CHUNKS_PKL)
    save_pickle(metas, META_PKL)

def ensure_dir():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
