"""src/retrieval/hybrid_searcher.py

In-process Hybrid Retriever combining FAISS (dense cosine) and BM25 (sparse keyword).
Follows standard linear score combination:
  final_score = alpha * norm_faiss + (1 - alpha) * norm_bm25

The searcher is designed so the LLM pipeline only passes query: str.
All internal search parameters (top_k, alpha, preset) are configured in code or config.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("/Users/adity/Repos/vaani/data/indexes/preset_512")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class HybridSearcher:
    """Loads pre-built FAISS and BM25 indexes from disk and executes hybrid queries."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        encoder_model_name: str = DEFAULT_MODEL_NAME,
        default_top_k: int = 5,
        default_alpha: float = 0.5,
    ):
        self.index_dir = Path(index_dir)
        self.top_k = default_top_k
        self.alpha = default_alpha

        logger.info("Initializing HybridSearcher from %s", self.index_dir)
        # 1. Load FAISS index from disk
        faiss_file = self.index_dir / "faiss.index"
        if not faiss_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_file}")
        self.faiss_index = faiss.read_index(str(faiss_file))

        # 2. Load BM25 model from disk
        bm25_file = self.index_dir / "bm25.pkl"
        if not bm25_file.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_file}")
        with open(bm25_file, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        # 3. Load chunk lookup manifest
        lookup_file = self.index_dir / "chunk_lookup.json"
        if not lookup_file.exists():
            raise FileNotFoundError(f"Chunk lookup not found: {lookup_file}")
        with open(lookup_file, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)

        # 4. Load lightweight query encoder
        self.encoder = SentenceTransformer(encoder_model_name)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize an array of scores into [0, 1] range."""
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min == 0:
            return np.ones_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search over indexed chunks.

        The LLM will only pass `query`. Optional parameters allow pipeline-level tuning.
        """
        k = top_k or self.top_k
        a = alpha if alpha is not None else self.alpha

        if not query.strip():
            return []

        # ---------------------------------------------------------
        # 1. DENSE SEARCH (FAISS)
        # ---------------------------------------------------------
        query_vec = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        # Retrieve all scores across corpus for normalization
        faiss_scores, faiss_indices = self.faiss_index.search(
            query_vec.astype(np.float32), len(self.chunks)
        )
        # faiss_scores is shape (1, N)
        faiss_all_scores = np.zeros(len(self.chunks))
        for rank, idx in enumerate(faiss_indices[0]):
            faiss_all_scores[idx] = faiss_scores[0][rank]

        # ---------------------------------------------------------
        # 2. SPARSE SEARCH (BM25)
        # ---------------------------------------------------------
        tokenized_query = [w.strip(".,;:!?()[]{}\"'`").lower() for w in query.split() if len(w) > 1]
        bm25_all_scores = np.array(self.bm25.get_scores(tokenized_query))

        # ---------------------------------------------------------
        # 3. SCORE NORMALIZATION & COMBINATION
        # ---------------------------------------------------------
        norm_faiss = self._normalize_scores(faiss_all_scores)
        norm_bm25 = self._normalize_scores(bm25_all_scores)

        # Linear mathematical combination: alpha * dense + (1 - alpha) * sparse
        hybrid_scores = (a * norm_faiss) + ((1.0 - a) * norm_bm25)

        # Get top-k chunk indices
        top_indices = np.argsort(-hybrid_scores)[:k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank + 1,
                "chunk_id": chunk["chunk_id"],
                "paper_title": chunk["paper_title"],
                "section": chunk["section"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "citation": chunk["citation"],
                "text": chunk["text"],
                "score": round(float(hybrid_scores[idx]), 4),
                "dense_score": round(float(norm_faiss[idx]), 4),
                "sparse_score": round(float(norm_bm25[idx]), 4),
            })

        return results

