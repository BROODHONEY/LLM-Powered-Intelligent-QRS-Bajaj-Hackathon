import os
from typing import List, Dict
from utils import read_url_text, sliding_window_chunks, timer_ms
from embedder import InstructorEmbedder
from FAISS.index_builder import build_index, ensure_dir
from FAISS.search_faiss import search
from dotenv import load_dotenv
from decision_engine.decision_engine_hf import generate_answer
load_dotenv()


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
# Use fewer retrieved chunks for faster prompting
TOP_K = int(os.getenv("TOP_K", "3"))
# Keep a smaller context to speed up token processing
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "1200"))

# Initialize a single embedder instance to avoid reloading per request
EMBEDDER_MODEL_NAME = os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDER = InstructorEmbedder(EMBEDDER_MODEL_NAME)

def _index_matches_document(document_link: str) -> Dict:
    """Return {matches: bool, num_chunks: int} if an index exists for the given document.
    We detect a match by checking the stored metas' 'source' field.
    """
    try:
        from FAISS.search_faiss import load_faiss
        index, chunks, metas = load_faiss()
        if metas and isinstance(metas, list):
            source = metas[0].get("source")
            if source == document_link:
                return {"matches": True, "num_chunks": len(chunks) if chunks else 0}
        return {"matches": False, "num_chunks": 0}
    except Exception:
        return {"matches": False, "num_chunks": 0}

def ingest_link_and_build(document_link: str) -> Dict:
    ensure_dir()
    total_start = timer_ms()

    # Fast path: if the same document is already indexed, skip rebuild
    cache_info = _index_matches_document(document_link)
    if cache_info.get("matches"):
        print("[INFO] Using cached FAISS index for the same document. Skipping ingestion.")
        print(f"[TIME] Ingestion total: {timer_ms() - total_start} ms (cached)")
        return {"num_chunks": cache_info.get("num_chunks", 0)}

    print("[INFO] Ingesting document...")
    t0 = timer_ms()
    text = read_url_text(document_link)
    print(f"[TIME] Fetch document: {timer_ms() - t0} ms")

    print("[INFO] Chunking text...")
    t1 = timer_ms()
    chunks = sliding_window_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"[TIME] Chunking: {timer_ms() - t1} ms (chunks={len(chunks)})")
    metas = [{"doc_id": 0, "chunk_id": i, "source": document_link} for i in range(len(chunks))]
    embedder = EMBEDDER

    print("[INFO] Initializing FAISS index...")
    t2 = timer_ms()
    build_index(chunks, metas, embedder, use_ivf=False)
    print(f"[TIME] Build index (embed + add + persist): {timer_ms() - t2} ms")
    print("[INFO] FAISS index built successfully.")

    print(f"[TIME] Ingestion total: {timer_ms() - total_start} ms")
    return {"num_chunks": len(chunks)}

def make_context(snippets: List[str]) -> str:
    # Simple concatenation with separators, enforce rough token limit by length
    ctx = ""
    for i, s in enumerate(snippets):
        # Always include the first snippet to avoid empty context
        if i > 0 and (len(ctx) + len(s) + 50 > MAX_CONTEXT_TOKENS * 4):  # crude char->token
            break
        ctx += f"\n\n[Source]\n{s}"
    return ctx.strip()

def answer_queries(queries: List[str]) -> List[str]:
    embedder = EMBEDDER
    answers = []
    from FAISS.search_faiss import load_faiss
    print(f"[INFO] Entering answer_queries. num_queries={len(queries)}", flush=True)
    index, chunks, metas = load_faiss()
    print("[INFO] FAISS index loaded for querying.", flush=True)
    for q in queries:
        print("[INFO] Searching chunks...", flush=True)
        t_search = timer_ms()
        results = search(q, embedder, k=TOP_K)
        print(f"[TIME] Retrieval: {timer_ms() - t_search} ms", flush=True)
        top_chunks = [chunks[idx] for idx, _ in results if idx >= 0]
        for idx, chunk in enumerate(top_chunks):
            print(f"[INFO] Found chunk {idx}: {chunk[:300]}", flush=True)

        t_ctx = timer_ms()
        context = make_context(top_chunks)
        print(f"[TIME] Build context: {timer_ms() - t_ctx} ms", flush=True)
        if not context:
            answers.append("I don't know.")
            continue
        prompt = (
            "You are a helpful assistant answering based on the provided context only. "
            "Cite facts concisely. If the answer is not in context, say you don't know.\n\n"
            f"Question: {q}\n\nContext:\n{context}\n\nAnswer: "
        )
        print("[INFO] Generating answer", flush=True)
        t_gen = timer_ms()
        # Generate fewer tokens for lower latency
        ans = generate_answer(prompt, max_tokens=int(os.getenv("MAX_NEW_TOKENS", "64")))
        print(f"[TIME] Generation (LLM): {timer_ms() - t_gen} ms", flush=True)
        print(f"[INFO] Generated answer: {ans.strip()}", flush=True)
        answers.append(ans.strip())
    return answers
