"""src/retrieval/extractor.py

Loosely coupled PDF extraction engine using Docling.
Preserves page numbers, section hierarchies, tables, and text elements
into structured JSON and annotated Markdown for RAG retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter

from src.retrieval.papers_meta import PAPERS

logger = logging.getLogger(__name__)


class DoclingExtractor:
    """Extracts structured document elements with full citation metadata (paper, section, page)."""

    def __init__(self):
        self.converter = DocumentConverter()

    def extract_paper(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Convert a PDF into elements.json and full_text.md with page and section metadata."""
        paper_key = pdf_path.stem
        paper_info = PAPERS.get(paper_key, {
            "title": pdf_path.stem,
            "arxiv_id": pdf_path.stem,
        })
        paper_title = paper_info["title"]

        output_dir.mkdir(parents=True, exist_ok=True)
        elements_file = output_dir / "elements.json"
        markdown_file = output_dir / "full_text.md"

        logger.info("Converting %s (%s)...", pdf_path.name, paper_title)
        conv_result = self.converter.convert(str(pdf_path))
        doc = conv_result.document

        elements: List[Dict[str, Any]] = []
        md_lines: List[str] = [
            f"# {paper_title}",
            f"*arXiv: {paper_key}*",
            "",
        ]

        current_section = "Frontmatter"
        current_page = 1
        item_counter = 0

        for item, level in doc.iterate_items():
            label = str(getattr(item, "label", "text")).lower()
            prov = getattr(item, "prov", None)
            page_no = prov[0].page_no if prov and len(prov) > 0 else current_page
            if page_no is not None:
                current_page = page_no

            text = ""
            elem_type = "text"

            if "section_header" in label or "title" in label:
                text = str(getattr(item, "text", "")).strip()
                if text:
                    current_section = text
                    elem_type = "section_header"
                    md_lines.append(f"\n<!-- [PAGE {current_page}] [SECTION: {current_section}] -->")
                    md_lines.append(f"{'#' * min(level + 1, 6)} {text}\n")
            elif "table" in label:
                elem_type = "table"
                # If table supports export_to_markdown, use that; otherwise fall back to text
                if hasattr(item, "export_to_markdown"):
                    text = item.export_to_markdown(doc=doc)
                else:
                    text = str(getattr(item, "text", "")).strip()
                if text:
                    md_lines.append(f"\n<!-- [PAGE {current_page}] [TABLE under {current_section}] -->\n")
                    md_lines.append(text + "\n")
            else:
                text = str(getattr(item, "text", "")).strip()
                if text:
                    elem_type = "text"
                    md_lines.append(text)

            if text:
                item_counter += 1
                elements.append({
                    "item_id": item_counter,
                    "paper_id": paper_key,
                    "paper_title": paper_title,
                    "section": current_section,
                    "page": current_page,
                    "type": elem_type,
                    "text": text,
                })

        # Save machine-readable JSON elements
        with open(elements_file, "w", encoding="utf-8") as f:
            json.dump(elements, f, indent=2, ensure_ascii=False)

        # Save human-readable annotated Markdown
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        summary = {
            "paper_id": paper_key,
            "paper_title": paper_title,
            "total_elements": len(elements),
            "total_pages": current_page,
            "elements_path": str(elements_file),
            "markdown_path": str(markdown_file),
        }
        return summary
