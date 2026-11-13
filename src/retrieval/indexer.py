"""src/retrieval/indexer.py

Builds and persists dual indexes for each chunk preset:
1. FAISS index: Dense semantic embeddings via sentence-transformers (all-MiniLM-L6-v2)
   with Inner Product (Cosine similarity when vectors are normalized).
2. BM25 index: Sparse keyword frequency via rank-bm25 (Okapi BM25).

Saved on disk per preset under:
  data/indexes/<preset_name>/
    ├── faiss.index
    ├── bm25.pkl
    └── chunk_lookup.json
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class HybridIndexer:
    """Creates and persists FAISS dense index and BM25 sparse index."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.encoder = SentenceTransformer(model_name)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """Simple, fast lowercase tokenization for BM25."""
        return [w.strip(".,;:!?()[]{}\"'`").lower() for w in text.split() if len(w) > 1]

    def build_indexes_for_preset(
        self,
        chunks_json_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Load chunks.json, build both FAISS and BM25, and persist to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks: List[Dict[str, Any]] = json.load(f)

        if not chunks:
            raise ValueError(f"No chunks found in {chunks_json_path}")

        logger.info("Building indexes for %d chunks from %s", len(chunks), chunks_json_path.name)
        corpus_texts = [c["text"] for c in chunks]

        # ---------------------------------------------------------
        # 1. BUILD FAISS DENSE INDEX
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        embeddings = self.encoder.encode(
            corpus_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalized vectors -> Inner Product == Cosine Similarity
        )
        embedding_dim = embeddings.shape[1]

        # Use IndexFlatIP (Inner Product) on normalized vectors for exact cosine similarity
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss_index.add(embeddings.astype(np.float32))

        faiss_file = output_dir / "faiss.index"
        faiss.write_index(faiss_index, str(faiss_file))
        faiss_time = time.perf_counter() - t0
        logger.info("FAISS index saved to %s (took %.2fs)", faiss_file.name, faiss_time)

        # ---------------------------------------------------------
        # 2. BUILD BM25 SPARSE INDEX
        # ---------------------------------------------------------
        t1 = time.perf_counter()
        tokenized_corpus = [self._tokenize_text(t) for t in corpus_texts]
        bm25_model = BM25Okapi(tokenized_corpus)

        bm25_file = output_dir / "bm25.pkl"
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25_model, f)
        bm25_time = time.perf_counter() - t1
        logger.info("BM25 index saved to %s (took %.2fs)", bm25_file.name, bm25_time)

        # ---------------------------------------------------------
        # 3. SAVE CHUNK LOOKUP MANIFEST
        # ---------------------------------------------------------
        # Mapping: integer index position in FAISS/BM25 -> full chunk object
        lookup_file = output_dir / "chunk_lookup.json"
        with open(lookup_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        summary = {
            "total_chunks": len(chunks),
            "embedding_dimension": embedding_dim,
            "faiss_file": str(faiss_file),
            "bm25_file": str(bm25_file),
            "lookup_file": str(lookup_file),
            "faiss_build_sec": round(faiss_time, 2),
            "bm25_build_sec": round(bm25_time, 2),
        }
        return summary

