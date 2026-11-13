"""src/pipeline/citation_resolver.py

Validates and resolves lightweight chunk citation tags [C1], [C2] into
exact canonical citations: [Paper Title, Section X, p. Y].
Detects citation hallucination (referencing non-existent chunks).
"""

import re
from typing import Any, Dict, List, Tuple


class CitationResolver:
    """Post-processor that verifies and formats chunk citations."""

    TAG_PATTERN = re.compile(r"\[C(\d+)\]")

    @classmethod
    def resolve(
        cls, answer: str, chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], bool, str]:
        """Resolves [C1], [C2] tags into canonical citations and validates grounding.

        Args:
            answer: Raw text from Drafter containing [C1], [C2], etc.
            chunks: The top-k chunks passed to the Drafter (1-indexed in prompt).

        Returns:
            Tuple of:
            - resolved_answer: Text with [C1] replaced by [Paper Title, Section X, p. Y]
            - citations: List of unique canonical citation strings cited in answer
            - is_valid: True if citations are valid; False if hallucinated tags or missing citations
            - message: Status/error message
        """
        if not answer.strip():
            return answer, [], True, "Empty answer"

        matches = cls.TAG_PATTERN.findall(answer)
        if not matches:
            # Answer provided but no citations found
            return (
                answer,
                [],
                False,
                "No chunk citations ([C1], [C2]...) found in the answer.",
            )

        num_chunks = len(chunks)
        cited_indices = [int(m) for m in matches]

        # Check for hallucinated chunk indices (e.g., citing [C5] when only 3 chunks were provided)
        invalid_indices = [idx for idx in cited_indices if idx < 1 or idx > num_chunks]
        if invalid_indices:
            return (
                answer,
                [],
                False,
                f"Hallucinated citation tags detected: {[f'[C{i}]' for i in invalid_indices]} "
                f"exceed available chunks (1..{num_chunks}).",
            )

        # Replace tags with canonical citations
        resolved_answer = answer
        unique_citations = []

        def replace_tag(match: re.Match) -> str:
            idx = int(match.group(1))
            canonical = chunks[idx - 1]["citation"]
            if canonical not in unique_citations:
                unique_citations.append(canonical)
            return canonical

        resolved_answer = cls.TAG_PATTERN.sub(replace_tag, answer)

        return resolved_answer, unique_citations, True, "Citations successfully verified and resolved."

