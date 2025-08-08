import os
from typing import List, Dict
from utils import read_url_text, sliding_window_chunks
from embedder import InstructorEmbedder
from FAISS.index_builder import build_index, ensure_dir
from FAISS.search_faiss import search
from decision_engine.decision_engine_hf import generate_answer
from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))

def ingest_link_and_build(document_link: str) -> Dict:
    ensure_dir()
    print("[INFO] Ingesting document...")
    text = read_url_text(document_link)

    print("[INFO] Chunking text...")
    chunks = sliding_window_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    metas = [{"doc_id": 0, "chunk_id": i, "source": document_link} for i in range(len(chunks))]
    embedder = InstructorEmbedder(os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    print("[INFO] Initializing FAISS index...")
    build_index(chunks, metas, embedder, use_ivf=False)
    print("[INFO] FAISS index built successfully.")

    return {"num_chunks": len(chunks)}

def make_context(snippets: List[str]) -> str:
    # Simple concatenation with separators, enforce rough token limit by length
    ctx = ""
    for s in snippets:
        if len(ctx) + len(s) + 50 > MAX_CONTEXT_TOKENS * 4:  # crude char->token
            break
        ctx += f"\n\n[Source]\n{s}"
    return ctx.strip()

def answer_queries(queries: List[str]) -> List[str]:
    embedder = InstructorEmbedder(os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    answers = []
    from FAISS.search_faiss import load_faiss
    index, chunks, metas = load_faiss()
    for q in queries:
        print("[INFO] Searching chunks...")
        results = search(q, embedder, k=TOP_K)
        top_chunks = [chunks[idx] for idx, _ in results if idx >= 0]
        for idx, chunk in enumerate(top_chunks):
            print(f"[INFO] Found chunk {idx}: {chunk[:300]}")

        context = make_context(top_chunks)
        prompt = (
            "You are a helpful assistant answering based on the provided context only. "
            "Cite facts concisely. If the answer is not in context, say you don't know.\n\n"
            f"Question: {q}\n\nContext:\n{context}\n\nAnswer: "
        )
        print("[INFO] Generating answer")
        ans = generate_answer(prompt)
        print(f"[INFO] Generated answer: {ans.strip()}")
        answers.append(ans.strip())
    return answers
