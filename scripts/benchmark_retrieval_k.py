"""scripts/benchmark_retrieval_k.py

Empirical evaluation:
1. Top-k=3 vs Top-k=5 accuracy / hit rate on 12 ground-truth questions across all 3 papers.
2. Context token load comparison.
3. Cross-Encoder re-ranking latency on Mac CPU.
4. Score distribution on in-domain vs 4 out-of-domain questions to calibrate the rejection threshold.

Uses:
- HybridSearcher (data/indexes/preset_512)
- CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import CrossEncoder

from src.retrieval.hybrid_searcher import HybridSearcher

logging.basicConfig(level=logging.WARNING)

# 12 In-Domain technical questions + ground truth validation criteria
# (Paper ID, Question, Must-contain keywords in top chunks to count as a Hit)
IN_DOMAIN_QUESTIONS = [
    # --- Attention Is All You Need ---
    {
        "id": "A1",
        "paper": "Attention Is All You Need",
        "query": "What are the dimensions d_model and d_k in the Transformer base model?",
        "expected_keywords": ["512", "64"],
    },
    {
        "id": "A2",
        "paper": "Attention Is All You Need",
        "query": "What optimizer and learning rate warmup schedule was used to train the Transformer?",
        "expected_keywords": ["adam", "warmup"],
    },
    {
        "id": "A3",
        "paper": "Attention Is All You Need",
        "query": "Why does the Transformer use multi-head attention instead of a single attention head?",
        "expected_keywords": ["multi-head", "subspaces"],
    },
    {
        "id": "A4",
        "paper": "Attention Is All You Need",
        "query": "What BLEU score did the big Transformer achieve on the WMT 2014 English-to-German task?",
        "expected_keywords": ["28.4", "bleu"],
    },
    # --- Moshi: speech-text foundation model ---
    {
        "id": "M1",
        "paper": "Moshi",
        "query": "What is Mimi neural audio codec and what is its framerate and bandwidth in Moshi?",
        "expected_keywords": ["mimi", "12.5"],
    },
    {
        "id": "M2",
        "paper": "Moshi",
        "query": "How does Moshi model multi-stream audio for full-duplex spoken dialogue?",
        "expected_keywords": ["stream", "full-duplex"],
    },
    {
        "id": "M3",
        "paper": "Moshi",
        "query": "What is the parameter size and architecture of the Helium text transformer in Moshi?",
        "expected_keywords": ["helium", "7b"],
    },
    {
        "id": "M4",
        "paper": "Moshi",
        "query": "What acoustic delay is introduced between audio tokens and text tokens in Moshi?",
        "expected_keywords": ["delay", "acoustic"],
    },
    # --- Dynamic Chunking ---
    {
        "id": "D1",
        "paper": "Dynamic Chunking",
        "query": "What is the key difference between fixed-size chunking and dynamic chunking?",
        "expected_keywords": ["dynamic", "fixed", "boundary"],
    },
    {
        "id": "D2",
        "paper": "Dynamic Chunking",
        "query": "How does the boundary predictor in dynamic chunking decide when to end a chunk?",
        "expected_keywords": ["boundary", "predictor"],
    },
    {
        "id": "D3",
        "paper": "Dynamic Chunking",
        "query": "What pooling mechanism is used to aggregate representations within a dynamic chunk?",
        "expected_keywords": ["pooling", "aggregate"],
    },
    {
        "id": "D4",
        "paper": "Dynamic Chunking",
        "query": "What benchmark or dataset was used to evaluate dynamic chunking sequence modeling?",
        "expected_keywords": ["benchmark", "lra"],
    },
]

# 4 Out-of-Domain questions to calibrate rejection threshold
OUT_OF_DOMAIN_QUESTIONS = [
    {
        "id": "O1",
        "type": "unrelated",
        "query": "How do you bake a sourdough bread with 75% hydration and open crumb?",
    },
    {
        "id": "O2",
        "type": "unrelated",
        "query": "What are the rules of chess regarding en passant capture and castling?",
    },
    {
        "id": "O3",
        "type": "adjacent_ml",
        "query": "How does Proximal Policy Optimization (PPO) compute clipped surrogate objectives in RLHF?",
    },
    {
        "id": "O4",
        "type": "adjacent_ml",
        "query": "What is the architecture of FlashAttention-3 forward kernel on Hopper GPUs with TMA?",
    },
]


def check_hit(chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> bool:
    """Check if any chunk contains all expected keywords."""
    combined_text = " ".join([c["text"].lower() for c in chunks])
    return all(kw.lower() in combined_text for kw in expected_keywords)


def count_tokens_approx(text: str) -> int:
    """Rough token count (~1.3 tokens per word)."""
    return int(len(text.split()) * 1.3)


def main():
    print("=" * 80)
    print("BENCHMARK: Top-k=3 vs Top-k=5 & Cross-Encoder Threshold Calibration")
    print("=" * 80)

    print("\n[1/4] Loading HybridSearcher (preset_512)...")
    searcher = HybridSearcher(index_dir=Path("data/indexes/preset_512"))

    print("\n[2/4] Loading CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    t0 = time.perf_counter()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"Loaded CrossEncoder in {time.perf_counter() - t0:.2f}s")

    # -------------------------------------------------------------
    # 3. Evaluate In-Domain Questions
    # -------------------------------------------------------------
    print("\n[3/4] Evaluating 12 In-Domain Questions...")
    in_results = []
    hybrid_times = []
    rerank_times = []

    for item in IN_DOMAIN_QUESTIONS:
        q = item["query"]
        expected = item["expected_keywords"]

        # Step 1: Hybrid Search (Candidate pool of 15 chunks)
        t_h0 = time.perf_counter()
        candidates = searcher.search(q, top_k=15, alpha=0.5)
        t_h = (time.perf_counter() - t_h0) * 1000
        hybrid_times.append(t_h)

        # Step 2: Cross-Encoder Reranking
        t_r0 = time.perf_counter()
        pairs = [[q, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        t_r = (time.perf_counter() - t_r0) * 1000
        rerank_times.append(t_r)

        # Attach scores and sort
        for i, score in enumerate(scores):
            candidates[i]["ce_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)

        top1_score = reranked[0]["ce_score"]
        hit_k1 = check_hit(reranked[:1], expected)
        hit_k3 = check_hit(reranked[:3], expected)
        hit_k5 = check_hit(reranked[:5], expected)

        tokens_k3 = sum(count_tokens_approx(c["text"]) for c in reranked[:3])
        tokens_k5 = sum(count_tokens_approx(c["text"]) for c in reranked[:5])

        in_results.append({
            "id": item["id"],
            "query": q[:45] + "...",
            "top1_score": top1_score,
            "hit_k1": hit_k1,
            "hit_k3": hit_k3,
            "hit_k5": hit_k5,
            "tokens_k3": tokens_k3,
            "tokens_k5": tokens_k5,
        })

    # -------------------------------------------------------------
    # 4. Evaluate Out-of-Domain Questions
    # -------------------------------------------------------------
    print("\n[4/4] Evaluating 4 Out-of-Domain Questions...")
    out_results = []
    for item in OUT_OF_DOMAIN_QUESTIONS:
        q = item["query"]
        candidates = searcher.search(q, top_k=15, alpha=0.5)
        pairs = [[q, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        top1_score = float(np.max(scores))
        out_results.append({
            "id": item["id"],
            "type": item["type"],
            "query": q[:45] + "...",
            "top1_score": top1_score,
        })

    # -------------------------------------------------------------
    # Summary Tables & Display
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("IN-DOMAIN RESULTS (Hit Rate & Scores)")
    print("=" * 80)
    print(f"{'ID':<4} | {'Top-1 Score':<11} | {'Hit@1':<5} | {'Hit@3':<5} | {'Hit@5':<5} | {'Tok@3':<5} | {'Tok@5':<5} | Query")
    print("-" * 80)
    for r in in_results:
        h1 = "✅" if r["hit_k1"] else "❌"
        h3 = "✅" if r["hit_k3"] else "❌"
        h5 = "✅" if r["hit_k5"] else "❌"
        print(f"{r['id']:<4} | {r['top1_score']:+11.4f} | {h1:<5} | {h3:<5} | {h5:<5} | {r['tokens_k3']:<5} | {r['tokens_k5']:<5} | {r['query']}")

    print("\n" + "=" * 80)
    print("OUT-OF-DOMAIN RESULTS (Rejection Calibration)")
    print("=" * 80)
    print(f"{'ID':<4} | {'Type':<12} | {'Top-1 Score':<11} | Query")
    print("-" * 80)
    for r in out_results:
        print(f"{r['id']:<4} | {r['type']:<12} | {r['top1_score']:+11.4f} | {r['query']}")

    # Aggregate Statistics
    hit_rate_k1 = sum(1 for r in in_results if r["hit_k1"]) / len(in_results) * 100
    hit_rate_k3 = sum(1 for r in in_results if r["hit_k3"]) / len(in_results) * 100
    hit_rate_k5 = sum(1 for r in in_results if r["hit_k5"]) / len(in_results) * 100

    avg_tok_k3 = np.mean([r["tokens_k3"] for r in in_results])
    avg_tok_k5 = np.mean([r["tokens_k5"] for r in in_results])

    in_scores = [r["top1_score"] for r in in_results]
    out_scores = [r["top1_score"] for r in out_results]

    min_in_score = min(in_scores)
    max_out_score = max(out_scores)

    print("\n" + "=" * 80)
    print("AGGREGATE COMPARISON")
    print("=" * 80)
    print(f"Hit Rate @ 1: {hit_rate_k1:.1f}%")
    print(f"Hit Rate @ 3: {hit_rate_k3:.1f}%")
    print(f"Hit Rate @ 5: {hit_rate_k5:.1f}%")
    print(f"Avg Context Tokens @ k=3: {avg_tok_k3:.0f} tokens (~{avg_tok_k3/7000*100:.1f}% of 7k ITPM)")
    print(f"Avg Context Tokens @ k=5: {avg_tok_k5:.0f} tokens (~{avg_tok_k5/7000*100:.1f}% of 7k ITPM)")
    print(f"Avg Hybrid Search Time:   {np.mean(hybrid_times):.1f} ms")
    print(f"Avg Cross-Encoder Time:   {np.mean(rerank_times):.1f} ms")
    print("-" * 80)
    print(f"In-Domain Top-1 Score Range:     [{min(in_scores):+.4f} to {max(in_scores):+.4f}] (mean: {np.mean(in_scores):+.4f})")
    print(f"Out-of-Domain Top-1 Score Range: [{min(out_scores):+.4f} to {max(out_scores):+.4f}] (mean: {np.mean(out_scores):+.4f})")
    print(f"Score Gap: Min In-Domain ({min_in_score:+.4f}) vs Max Out-of-Domain ({max_out_score:+.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()

