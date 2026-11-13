"""src/retrieval/chunker.py

Element-aware and section-aware text chunker with provenance tracking.
Assembles clean Docling elements into overlapping passages of target token sizes.
Generates 3 required presets: 256, 512, and 1024 tokens.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Use a standard, fast HuggingFace tokenizer (GPT-2 / tiktoken style) for realistic token counts
TOKENIZER_NAME = "gpt2"


class DocumentChunker:
    """Chunks structured document elements while preserving section and page provenance."""

    def __init__(self, tokenizer_name: str = TOKENIZER_NAME):
        logger.info("Initializing tokenizer: %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_paper(
        self,
        cleaned_elements: List[Dict[str, Any]],
        target_tokens: int,
        overlap_tokens: int,
        preset_name: str,
    ) -> List[Dict[str, Any]]:
        """Group elements into chunks of target_tokens with overlap_tokens.

        Respects:
        - Section boundaries (flushes when moving across distinct major sections).
        - Tables (kept as atomic blocks).
        - Page ranges (tracks page_start and page_end).
        """
        chunks: List[Dict[str, Any]] = []
        if not cleaned_elements:
            return chunks

        paper_id = cleaned_elements[0]["paper_id"]
        paper_title = cleaned_elements[0]["paper_title"]

        accumulated_elements: List[Dict[str, Any]] = []
        accumulated_text_parts: List[str] = []
        current_tokens = 0
        current_section = ""
        chunk_idx = 0

        def build_chunk(elements_slice: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            nonlocal chunk_idx
            if not elements_slice:
                return None

            chunk_idx += 1
            combined_text = "\n\n".join(e["text"] for e in elements_slice if e["text"])
            token_count = self.count_tokens(combined_text)

            pages = [e["page"] for e in elements_slice if e.get("page")]
            page_start = min(pages) if pages else 1
            page_end = max(pages) if pages else page_start

            section_names = [e["section"] for e in elements_slice if e.get("section")]
            section_display = section_names[0] if section_names else "General"

            # Pre-formatted citation string
            if page_start == page_end:
                page_str = f"p. {page_start}"
            else:
                page_str = f"p. {page_start}-{page_end}"

            citation = f"[{paper_title}, Section {section_display}, {page_str}]"
            chunk_id = f"{paper_id}_{target_tokens}_{chunk_idx:04d}"

            return {
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section": section_display,
                "page_start": page_start,
                "page_end": page_end,
                "token_count": token_count,
                "element_count": len(elements_slice),
                "has_table": any(e.get("type") == "table" for e in elements_slice),
                "text": combined_text,
                "citation": citation,
            }

        for elem in cleaned_elements:
            elem_text = elem["text"]
            elem_tokens = self.count_tokens(elem_text)
            elem_section = elem.get("section", "")

            # Check if section changed to a completely different major section
            is_new_major_section = (
                current_section
                and elem_section
                and elem_section != current_section
                and elem.get("type") == "section_header"
            )

            # If adding this element exceeds target size OR a major section boundary is hit
            if (current_tokens + elem_tokens > target_tokens and accumulated_elements) or is_new_major_section:
                new_chunk = build_chunk(accumulated_elements)
                if new_chunk and new_chunk["token_count"] > 0:
                    chunks.append(new_chunk)

                # Calculate overlap: keep recent elements that fit within overlap_tokens
                if is_new_major_section:
                    # Flush completely across major section headers
                    accumulated_elements = []
                    current_tokens = 0
                else:
                    overlap_elements: List[Dict[str, Any]] = []
                    overlap_accum = 0
                    for rev_elem in reversed(accumulated_elements):
                        rev_tokens = self.count_tokens(rev_elem["text"])
                        if overlap_accum + rev_tokens <= overlap_tokens:
                            overlap_elements.insert(0, rev_elem)
                            overlap_accum += rev_tokens
                        else:
                            break
                    accumulated_elements = overlap_elements
                    current_tokens = overlap_accum

            accumulated_elements.append(elem)
            current_tokens += elem_tokens
            if elem_section:
                current_section = elem_section

        # Final remaining elements
        if accumulated_elements:
            final_chunk = build_chunk(accumulated_elements)
            if final_chunk and final_chunk["token_count"] > 0:
                chunks.append(final_chunk)

        return chunks


def save_chunks_preset(
    chunks: List[Dict[str, Any]],
    output_dir: Path,
    preset_name: str,
) -> Dict[str, Any]:
    """Save chunks to chunks.json and manifest.csv under the specified preset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "chunks.json"
    csv_path = output_dir / "manifest.csv"

    # Save complete JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Save flat CSV manifest for easy inspection
    fieldnames = [
        "chunk_id",
        "paper_id",
        "paper_title",
        "section",
        "page_start",
        "page_end",
        "token_count",
        "has_table",
        "citation",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for c in chunks:
            writer.writerow(c)

    summary = {
        "preset": preset_name,
        "total_chunks": len(chunks),
        "avg_tokens": round(sum(c["token_count"] for c in chunks) / max(len(chunks), 1), 1),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
    return summary

