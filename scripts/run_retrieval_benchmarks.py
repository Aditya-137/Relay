"""scripts/run_retrieval_benchmarks.py

Runs the 15-question benchmark across 4 distinct retrieval settings to populate Table 3
("Retrieval: quality versus cost") in the final assignment report:

Setting 1: preset_256,  top_k=3, rerank=YES (Small chunks)
Setting 2: preset_512,  top_k=3, rerank=YES (Medium chunks - Primary Pipeline)
Setting 3: preset_512,  top_k=3, rerank=NO  (Medium chunks - Pure Math Hybrid)
Setting 4: preset_1024, top_k=3, rerank=YES (Large chunks)

Each question is asked by a distinct engineer user_id to evaluate retrieval quality cleanly.
All results are logged to results/results.csv with complete latency and token telemetry.
"""

import logging
import sys
import time
from pathlib import Path

from src.pipeline.orchestrator import VaaniPipeline

logging.basicConfig(level=logging.WARNING)

BENCHMARK_QUESTIONS = [
    # --- Attention Is All You Need ---
    {"id": "A1", "query": "What are the dimensions d_model and d_k in the Transformer base model?"},
    {"id": "A2", "query": "What optimizer and learning rate warmup schedule was used to train the Transformer?"},
    {"id": "A3", "query": "Why does the Transformer use multi-head attention instead of a single attention head?"},
    {"id": "A4", "query": "What BLEU score did the big Transformer achieve on the WMT 2014 English-to-German task?"},
    # --- Moshi ---
    {"id": "M1", "query": "What is Mimi neural audio codec and what is its framerate and bandwidth in Moshi?"},
    {"id": "M2", "query": "How does Moshi model multi-stream audio for full-duplex spoken dialogue?"},
    {"id": "M3", "query": "What is the parameter size and architecture of the Helium text transformer in Moshi?"},
    {"id": "M4", "query": "What acoustic delay is introduced between audio tokens and text tokens in Moshi?"},
    # --- Dynamic Chunking ---
    {"id": "D1", "query": "What is the key difference between fixed-size chunking and dynamic chunking?"},
    {"id": "D2", "query": "How does the boundary predictor in dynamic chunking decide when to end a chunk?"},
    {"id": "D3", "query": "What pooling mechanism is used to aggregate representations within a dynamic chunk?"},
    {"id": "D4", "query": "What benchmark or dataset was used to evaluate dynamic chunking sequence modeling?"},
    # --- Out-of-Domain ---
    {"id": "O1", "query": "How do you bake sourdough bread with 75% hydration and open crumb?"},
    {"id": "O2", "query": "What are the rules of chess regarding en passant capture and castling?"},
    # --- Vague / Conversational ---
    {"id": "V1", "query": "How big is the conversational speech model and how fast does its audio tokenizer run?"},
]

SETTINGS = [
    {"name": "Setting 1 (256, k=3, rerank=yes)", "preset": "preset_256", "top_k": 3, "use_reranker": True},
    {"name": "Setting 2 (512, k=3, rerank=yes)", "preset": "preset_512", "top_k": 3, "use_reranker": True},
    {"name": "Setting 3 (512, k=3, rerank=no)",  "preset": "preset_512", "top_k": 3, "use_reranker": False},
    {"name": "Setting 4 (1024, k=3, rerank=yes)", "preset": "preset_1024", "top_k": 3, "use_reranker": True},
]


def main():
    print("=" * 80)
    print("TABLE 3 RETRIEVAL BENCHMARK: 15 Questions x 4 Settings")
    print("=" * 80)

    pipeline = VaaniPipeline()

    total_runs = len(SETTINGS) * len(BENCHMARK_QUESTIONS)
    run_idx = 0

    for s_idx, setting in enumerate(SETTINGS, 1):
        print(f"\n[{s_idx}/4] RUNNING {setting['name'].upper()}...")
        print("-" * 80)

        for q_idx, item in enumerate(BENCHMARK_QUESTIONS, 1):
            run_idx += 1
            qid = item["id"]
            q = item["query"]

            t0 = time.perf_counter()
            res = pipeline.ask(
                query=q,
                user_id=f"eval_user_{s_idx}_{qid}",  # Distinct user per question
                scenario="table3_retrieval_benchmark",
                source="real",
                request_id=f"t3-s{s_idx}-{qid}",
                custom_preset=setting["preset"],
                custom_top_k=setting["top_k"],
                use_reranker=setting["use_reranker"],
            )
            elapsed = time.perf_counter() - t0

            status_symbol = "✅" if res.status == "success" else ("🚫" if "refused" in res.status else "⚠️")
            print(
                f"  [{run_idx:02d}/{total_runs:02d}] {status_symbol} {qid} ({res.status}): "
                f"TTFT={res.latency.ttft_ms or 0:.0f}ms | Total={res.latency.total_ms:.0f}ms | "
                f"Tokens In={res.token_usage.input_tokens} Out={res.token_usage.output_tokens} | "
                f"Citations={len(res.citations)}"
            )

            # 3.5s sleep to stay well within global 30 RPM / 7k ITPM
            time.sleep(3.5)

    print("\n" + "=" * 80)
    print("✅ TABLE 3 BENCHMARK COMPLETE! All 60 rows saved to results/results.csv.")
    print("=" * 80)


if __name__ == "__main__":
    main()
