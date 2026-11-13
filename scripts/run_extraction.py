"""scripts/run_extraction.py

Batch extraction runner for the three research papers.
Saves structured elements.json and full_text.md under data/extracted/<paper_id>/
"""

import json
import logging
import time
from pathlib import Path

from src.retrieval.extractor import DoclingExtractor
from src.retrieval.papers_meta import PAPERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PAPERS_DIR = Path("/Users/adity/Repos/vaani/data/papers")
EXTRACTED_DIR = Path("/Users/adity/Repos/vaani/data/extracted")


def main():
    extractor = DoclingExtractor()
    manifest = {}
    overall_start = time.perf_counter()

    for paper_id, info in PAPERS.items():
        pdf_path = PAPERS_DIR / info["file_name"]
        if not pdf_path.exists():
            logger.error("File not found: %s", pdf_path)
            continue

        paper_output_dir = EXTRACTED_DIR / paper_id
        logger.info("=" * 60)
        logger.info("Starting extraction for: %s (%s)", info["title"], paper_id)
        start_t = time.perf_counter()

        summary = extractor.extract_paper(pdf_path, paper_output_dir)
        elapsed = time.perf_counter() - start_t
        summary["duration_seconds"] = round(elapsed, 2)
        manifest[paper_id] = summary

        logger.info(
            "Extracted %s in %.2fs: %d elements across %d pages",
            paper_id,
            elapsed,
            summary["total_elements"],
            summary["total_pages"],
        )

    # Save manifest of all extracted papers
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = EXTRACTED_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.perf_counter() - overall_start
    logger.info("=" * 60)
    logger.info("All extractions completed in %.2fs. Manifest written to %s", total_time, manifest_file)


if __name__ == "__main__":
    main()

