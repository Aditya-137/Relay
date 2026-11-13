"""scripts/smoke_test_pipeline.py

Smoke test the end-to-end Vaani pipeline on:
1. In-domain question (Attention Is All You Need)
2. In-domain question (Moshi)
3. Out-of-domain question (Sourdough baking)
"""

import json
import logging
from src.pipeline.orchestrator import VaaniPipeline

logging.basicConfig(level=logging.INFO)

TEST_QUERIES = [
    "What are the dimensions d_model and d_k in the Transformer base model?",
    "What is Mimi in Moshi and what framerate and bandwidth does it operate at?",
    "How do you bake sourdough bread with 75% hydration?",
]


def main():
    print("=" * 80)
    print("VAANI PIPELINE END-TO-END SMOKE TEST")
    print("=" * 80)

    pipeline = VaaniPipeline()

    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/3] QUERY: {q}")
        print("-" * 80)
        res = pipeline.ask(q)

        print(f"Status:        {res.status}")
        print(f"Is Answerable: {res.is_answerable}")
        print(f"TTFT:          {res.latency.ttft_ms} ms")
        print(f"Total Time:    {res.latency.total_ms} ms (Working: {res.latency.working_ms}ms, Waiting: {res.latency.waiting_ms}ms, Retrying: {res.latency.retrying_ms}ms)")
        print(f"Tokens Used:   Input: {res.token_usage.input_tokens}, Output: {res.token_usage.output_tokens}")
        print(f"Citations:     {res.citations}")
        print(f"Answer:\n{res.answer}")
        print("-" * 80)


if __name__ == "__main__":
    main()

