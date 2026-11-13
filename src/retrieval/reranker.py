"""src/retrieval/reranker.py

In-process local Cross-Encoder reranker.
Scores (query, chunk_text) pairs using a local transformer (0 Groq API tokens).
"""

import logging
from typing import Any, Dict, List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Local CrossEncoder that scores and sorts candidate chunks."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        logger.info("Initializing CrossEncoderReranker with %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Rerank candidates based on Cross-Encoder score against the query.

        Args:
            query: The user search query.
            candidates: List of chunk dictionaries from HybridSearcher.
            top_k: Number of top chunks to return.

        Returns:
            Tuple of (top_k reranked candidates with 'rerank_score' attached, top1_score).
        """
        if not candidates or not query.strip():
            return [], -999.0

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = round(float(score), 4)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        top1_score = reranked[0]["rerank_score"] if reranked else -999.0

        return reranked[:top_k], top1_score

