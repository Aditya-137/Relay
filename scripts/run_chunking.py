"""scripts/run_chunking.py

Runner script that cleans elements and generates the 3 required chunking presets:
- preset_256 (Small: ~256 tokens, 50 token overlap)
- preset_512 (Medium: ~512 tokens, 100 token overlap)
- preset_1024 (Large: ~1024 tokens, 150 token overlap)
"""

import json
import logging
from pathlib import Path

from src.retrieval.chunker import DocumentChunker, save_chunks_preset
from src.retrieval.cleaner import clean_elements
from src.retrieval.papers_meta import PAPERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXTRACTED_DIR = Path("/Users/adity/Repos/vaani/data/extracted")
CHUNKS_DIR = Path("/Users/adity/Repos/vaani/data/chunks")

# Define the 3 required settings from the assignment brief
PRESETS = {
    "preset_256": {"target_tokens": 256, "overlap_tokens": 50},
    "preset_512": {"target_tokens": 512, "overlap_tokens": 100},
    "preset_1024": {"target_tokens": 1024, "overlap_tokens": 150},
}


def main():
    chunker = DocumentChunker()
    all_cleaned_papers = {}

    # Step 1: Clean elements for each paper
    logger.info("=" * 60)
    logger.info("STEP 1: Cleaning raw extracted elements")
    logger.info("=" * 60)
    for paper_id, info in PAPERS.items():
        elements_file = EXTRACTED_DIR / paper_id / "elements.json"
        if not elements_file.exists():
            logger.error("Missing elements file: %s", elements_file)
            continue

        with open(elements_file, "r", encoding="utf-8") as f:
            raw_elements = json.load(f)

        cleaned = clean_elements(raw_elements)
        all_cleaned_papers[paper_id] = cleaned
        logger.info(
            "Cleaned %s (%s): %d -> %d elements (filtered %d)",
            paper_id,
            info["title"][:30],
            len(raw_elements),
            len(cleaned),
            len(raw_elements) - len(cleaned),
        )

    # Step 2: Generate each chunk preset across all papers
    logger.info("=" * 60)
    logger.info("STEP 2: Generating 3 Chunk Presets")
    logger.info("=" * 60)

    summary_manifest = {}
    for preset_name, config in PRESETS.items():
        target_tokens = config["target_tokens"]
        overlap_tokens = config["overlap_tokens"]
        preset_output_dir = CHUNKS_DIR / preset_name

        combined_chunks = []
        for paper_id, elements in all_cleaned_papers.items():
            paper_chunks = chunker.chunk_paper(
                elements,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                preset_name=preset_name,
            )
            combined_chunks.extend(paper_chunks)

        preset_summary = save_chunks_preset(combined_chunks, preset_output_dir, preset_name)
        preset_summary["target_tokens"] = target_tokens
        preset_summary["overlap_tokens"] = overlap_tokens
        summary_manifest[preset_name] = preset_summary

        logger.info(
            "[%s] Target: %d tokens | Produced: %d chunks | Avg tokens: %.1f",
            preset_name,
            target_tokens,
            preset_summary["total_chunks"],
            preset_summary["avg_tokens"],
        )

    # Save summary of all presets
    with open(CHUNKS_DIR / "presets_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("Chunking complete! Summary written to data/chunks/presets_summary.json")


if __name__ == "__main__":
    main()

