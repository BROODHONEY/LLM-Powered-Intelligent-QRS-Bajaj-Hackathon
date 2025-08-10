import os
import re
from collections import Counter
from typing import List, Dict
from utils import read_url_text, sliding_window_chunks, timer_ms
from embedder import InstructorEmbedder
from FAISS.index_builder import build_index, ensure_dir
from FAISS.search_faiss import search
from dotenv import load_dotenv
from decision_engine.decision_engine_hf import generate_answer
load_dotenv()


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# Number of top chunks to retrieve and use for context
TOP_K = int(os.getenv("TOP_K", "12"))  # Increased from 8 to 12 for better coverage
# Keep a smaller context to speed up token processing
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2500"))  # Increased from 1800 to 2500 for better coverage
# Number of FAISS candidates per query before reranking (configurable; no content-based branching)
CANDIDATE_MULTIPLIER = int(os.getenv("CANDIDATE_MULTIPLIER", "20"))  # Increased from 15 to 20

# Initialize a single embedder instance to avoid reloading per request
# Default to a strong retriever: intfloat/e5-base
EMBEDDER_MODEL_NAME = os.getenv("EMBEDDER_MODEL", "intfloat/e5-base")
EMBEDDER = InstructorEmbedder(EMBEDDER_MODEL_NAME)

def _index_matches_document(document_link: str) -> Dict:
    """Return {matches: bool, num_chunks: int} if an index exists for the given document.
    We detect a match by checking the stored metas' 'source' field.
    """
    try:
        from FAISS.search_faiss import load_faiss
        index, chunks, metas = load_faiss()
        if metas and isinstance(metas, list):
            m0 = metas[0] if isinstance(metas[0], dict) else {}
            source = m0.get("source")
            emb_model = m0.get("embedding_model")
            m_chunk_size = m0.get("chunk_size")
            m_overlap = m0.get("chunk_overlap")
            if (
                source == document_link
                and emb_model == EMBEDDER_MODEL_NAME
                and m_chunk_size == CHUNK_SIZE
                and m_overlap == CHUNK_OVERLAP
            ):
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
    metas = [
        {
            "doc_id": 0,
            "chunk_id": i,
            "source": document_link,
            "embedding_model": EMBEDDER_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        for i in range(len(chunks))
    ]
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
        ctx += f"\n\n[Chunk {i+1}]\n{s}"
    return ctx.strip()

# --- Simple keyword-based reranker to improve accuracy ---
STOPWORDS = set(
    """
    a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with from your you we us our
    """.split()
)

def _normalize_words(text: str) -> Counter:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    # Keep short but meaningful tokens like "si", "ppn", etc. (len >= 2)
    return Counter(w for w in words if len(w) >= 2 and w not in STOPWORDS)

def _has_numbers(text: str) -> bool:
    return re.search(r"\b\d+(?:[.,]\d+)?\b", text) is not None

def rerank_indices(question: str, results, chunks, take_k: int) -> list:
    # results: List[(idx, score)] from FAISS (higher is better)
    # Expand candidates and rerank by overlap + numeric bias
    q_tokens = _normalize_words(question)
    # Choose up to 4 strongest keywords as guidance terms
    must_have = set([w for w, _ in q_tokens.most_common(4)])
    max_faiss = max((s for _, s in results), default=1.0) or 1.0
    
    scored = []
    for idx, s in results:
        if idx < 0:
            continue
        ch = chunks[idx]
        ch_tokens = _normalize_words(ch)
        
        # Overlap
        overlap_count = sum((ch_tokens & q_tokens).values())
        overlap_norm = overlap_count / max(1, len(q_tokens))
        
        # Numeric bonus
        numeric_bonus = 1.0 if (_has_numbers(question) and _has_numbers(ch)) else 0.0
        
        # Must-have keyword hits
        hits = len(must_have.intersection(set(ch_tokens.keys())))
        hit_ratio = hits / max(1, len(must_have))
        
        # FAISS score normalization
        faiss_part = s / max_faiss
        
        # More balanced scoring - don't penalize chunks too heavily for low overlap
        # This ensures we don't miss chunks that might contain the answer
        total = (0.35 * faiss_part + 
                 0.25 * overlap_norm + 
                 0.25 * hit_ratio + 
                 0.15 * numeric_bonus)
        
        scored.append((total, idx))
    
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:take_k]]


_sentence_split_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def _split_sentences(text: str) -> List[str]:
    parts = _sentence_split_re.split(text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 1]

def select_relevant_sentences(question: str, texts: List[str], max_sentences: int = 10) -> List[str]:
    q_tokens = _normalize_words(question)
    scored: List[tuple] = []
    
    
    for t in texts:
        for s in _split_sentences(t):
            s_tokens = _normalize_words(s)
            overlap = sum((s_tokens & q_tokens).values())
            overlap_norm = overlap / max(1, len(q_tokens))
            numeric_bonus = 0.15 if (_has_numbers(question) and _has_numbers(s)) else 0.0
            score = overlap_norm + numeric_bonus
            # Don't filter out sentences with 0 score - they might contain important info
            scored.append((score, s))
    
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    
    top = []
    seen = set()
    for _, s in scored:
        if s in seen:
            continue
        seen.add(s)
        top.append(s)
        if len(top) >= max_sentences:
            break
    
    
    if len(top) < max_sentences and len(texts) > 0:
        remaining_sentences = []
        for t in texts:
            sentences = _split_sentences(t)
            for s in sentences:
                if s not in seen and len(s.strip()) > 10:  # Only meaningful sentences
                    remaining_sentences.append(s)
                    seen.add(s)
        
        
        for s in remaining_sentences:
            if len(top) >= max_sentences:
                break
            top.append(s)
    
    
    if len(top) < max_sentences // 2:  
        print(f"[INFO] Very few sentences found ({len(top)}), expanding to include more from each chunk", flush=True)
        for t in texts:
            sentences = _split_sentences(t)
            
            for i, s in enumerate(sentences[:3]):
                if s not in seen and len(top) < max_sentences:
                    top.append(s)
                    seen.add(s)
    
    return top

def answer_queries(queries: List[str]) -> List[str]:
    embedder = EMBEDDER
    answers = []
    from FAISS.search_faiss import load_faiss
    print(f"[INFO] Entering answer_queries. num_queries={len(queries)}", flush=True)
    index,chunks,metas = load_faiss()
    print("[INFO] FAISS index loaded for querying.", flush=True)
    for q in queries:
        print("[INFO] Searching chunks...", flush=True)
        t_search = timer_ms()
        
        faiss_candidates = max(TOP_K * CANDIDATE_MULTIPLIER, TOP_K)
        results = search(q, embedder, k=faiss_candidates)
        print(f"[TIME] Retrieval: {timer_ms() - t_search} ms", flush=True)
        chosen_indices = rerank_indices(q, results, chunks, TOP_K)
        top_chunks = [chunks[idx] for idx in chosen_indices]
        for idx, chunk in enumerate(top_chunks):
            print(f"[INFO] Found chunk {idx}: {chunk[:300]}", flush=True)

        t_ctx = timer_ms()
        
        max_sent = min(25, TOP_K * 5)  
        best_sentences = select_relevant_sentences(q, top_chunks, max_sentences=max_sent)
        
        
        if len(best_sentences) < 10:  
            print(f"[INFO] Only {len(best_sentences)} sentences found, expanding context...", flush=True)
            
            for c in top_chunks:
                first_sentences = _split_sentences(c)
                if first_sentences:
                    first_sent = first_sentences[0]
                    if first_sent not in best_sentences and len(best_sentences) < max_sent:
                        best_sentences.append(first_sent)
        
        
        if len(best_sentences) < 15:
            print(f"[INFO] Still only {len(best_sentences)} sentences, adding more from each chunk", flush=True)
            for c in top_chunks:
                sentences = _split_sentences(c)
                for i, s in enumerate(sentences[1:3]):  
                    if s not in best_sentences and len(best_sentences) < max_sent and len(s.strip()) > 15:
                        best_sentences.append(s)
        
        
        if len(best_sentences) < 8:
            print("[WARN] Sentence selection yielded too few sentences, using full chunks.", flush=True)
            context_source = top_chunks
        else:
            context_source = best_sentences
            
        context = make_context(context_source)
        print(f"[TIME] Build context: {timer_ms() - t_ctx} ms", flush=True)
        print(f"[INFO] Context built with {len(context_source)} sources, length: {len(context)}", flush=True)
        
        if not context:
            answers.append("I don't know.")
            continue
            
        prompt = (
            "You are a precise question-answering assistant. Use ONLY the provided Context. "
            "Carefully analyze ALL the provided context to find the answer. "
            "Be thorough and look for information that might be spread across different chunks. "
            "If the answer is not explicitly present in the Context, reply exactly: I don't know. "
            "Do not use outside knowledge.\n\n"
            f"Question: {q}\n\n"
            "Context (each [Chunk N] is an independent excerpt):\n"
            f"{context}\n\n"
            "Instructions:\n"
            "- Respond with a concise but complete answer in 1–2 sentences.\n"
            "- Include key conditions and limits quoted exactly from the Context (numbers/percentages/units).\n"
            "- Do not add citations or restate the question.\n"
            "- Look for answers across ALL chunks - information might be spread out.\n"
            "- Pay special attention to waiting periods, limits, and specific conditions.\n"
            "- If the Context does not contain the answer, reply exactly: I don't know.\n\n"
            "Answer: "
        )
        print("[INFO] Generating answer", flush=True)
        t_gen = timer_ms()
        
        raw = generate_answer(prompt, max_tokens=int(os.getenv("MAX_NEW_TOKENS", "128")))
        
        text = raw.strip()
        
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = " ".join(lines) if lines else text
        
        text = re.sub(r"^answer\s*:\s*", "", text, flags=re.I)
        text = re.sub(r"\s*\[chunk\s*\d+[^\]]*\]\s*", "", text, flags=re.I)
        lowered = text.lower()
        if "i don't know" in lowered or "i do not know" in lowered:
            ans = "I don't know."
        else:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            kept = [s.strip() for s in sentences[:2] if s.strip()]
            if not kept:
                kept = [text.split(";")[0].split("  ")[0].strip()]
            ans = re.sub(r"\s+", " ", " ".join(kept)).strip()
        print(f"[TIME] Generation (LLM): {timer_ms() - t_gen} ms", flush=True)
        print(f"[INFO] Generated answer: {ans.strip()}", flush=True)
        answers.append(ans.strip())
    return answers
