"""scripts/run_indexing.py

Batch index builder for all 3 chunk presets:
- preset_256
- preset_512
- preset_1024

Saves faiss.index, bm25.pkl, and chunk_lookup.json under data/indexes/<preset>/
"""

import json
import logging
import time
from pathlib import Path

from src.retrieval.indexer import HybridIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("/Users/adity/Repos/vaani/data/chunks")
INDEXES_DIR = Path("/Users/adity/Repos/vaani/data/indexes")


def main():
    indexer = HybridIndexer()
    summary = {}
    total_start = time.perf_counter()

    for preset_dir in sorted(CHUNKS_DIR.glob("preset_*")):
        chunks_file = preset_dir / "chunks.json"
        if not chunks_file.exists():
            continue

        preset_name = preset_dir.name
        output_dir = INDEXES_DIR / preset_name
        logger.info("=" * 60)
        logger.info("Building indexes for preset: %s", preset_name)

        res = indexer.build_indexes_for_preset(chunks_file, output_dir)
        summary[preset_name] = res

    # Save overall summary
    with open(INDEXES_DIR / "indexing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_time = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info("All indexes built in %.2fs! Summary in data/indexes/indexing_summary.json", total_time)


if __name__ == "__main__":
    main()

