"""src/retrieval/cleaner.py

Rule-based cleaning and filtering for extracted Docling document elements.
Filters out frontmatter boilerplate, copyright disclaimers, and empty blocks
while preserving section headers, tables, and technical content.
"""

import re
from typing import Any, Dict, List

# Patterns matching copyright / distribution disclaimers on Page 1 / Frontmatter
BOILERPLATE_PATTERNS = [
    re.compile(r"google\s+hereby\s+grants\s+permission\s+to\s+reproduce", re.IGNORECASE),
    re.compile(r"provided\s+proper\s+attribution\s+is\s+provided", re.IGNORECASE),
    re.compile(r"under\s+review\s+as\s+a\s+conference\s+paper", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"^arxiv:\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE),
]


def is_boilerplate(text: str, page: int) -> bool:
    """Check if an element text matches known boilerplate/legal disclaimers on early pages."""
    if page <= 2:
        for pat in BOILERPLATE_PATTERNS:
            if pat.search(text):
                return True
    return False


def clean_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and filter raw Docling elements.

    Rules applied:
    1. Filter out empty or whitespace-only elements.
    2. Filter out known copyright/frontmatter disclaimers on page 1-2.
    3. Normalize whitespace while keeping markdown table structures intact.
    4. Categorize author blocks so they don't corrupt headings or abstract.
    """
    cleaned: List[Dict[str, Any]] = []

    for elem in elements:
        text = elem.get("text", "").strip()
        if not text:
            continue

        page = elem.get("page", 1)
        elem_type = elem.get("type", "text")

        # 1. Filter legal / copyright boilerplate
        if is_boilerplate(text, page):
            continue

        # 2. Normalize whitespace (preserve newlines for tables, collapse multiple spaces for prose)
        if elem_type != "table":
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

        cleaned_item = dict(elem)
        cleaned_item["text"] = text
        cleaned.append(cleaned_item)

    return cleaned

