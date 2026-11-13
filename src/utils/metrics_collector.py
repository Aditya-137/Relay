"""src/utils/metrics_collector.py

Collects per-request execution metrics and appends them to results/results.csv.
Provides the exact columns needed to compute Tables 2a, 2b, and 3 for the final report.
"""

import csv
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "results.csv"

FIELDNAMES = [
    "request_id",
    "timestamp",
    "user_id",
    "scenario",
    "source",
    "status",
    "ttft_ms",
    "total_ms",
    "waiting_ms",
    "working_ms",
    "retrying_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "retries_count",
    "retrieval_setting",
    "is_degraded",
    "citations_count",
    "citations",
    "query",
    "answer_snippet",
]


class MetricsCollector:
    """Thread-safe CSV logger for request-level telemetry."""

    def __init__(self, csv_path: Path = DEFAULT_CSV_PATH):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self):
        """Creates the CSV file with headers if it doesn't exist."""
        with self.lock:
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()

    def log_request(
        self,
        request_id: str,
        query: str,
        status: str,
        user_id: str = "default_user",
        scenario: str = "interactive",
        source: str = "real",
        retrieval_setting: str = "preset_512_k3",
        ttft_ms: Optional[float] = None,
        total_ms: float = 0.0,
        waiting_ms: float = 0.0,
        working_ms: float = 0.0,
        retrying_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retries_count: int = 0,
        is_degraded: bool = False,
        citations: Optional[List[str]] = None,
        answer: str = "",
    ):
        citations = citations or []
        row = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "scenario": scenario,
            "source": source,
            "status": status,
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else "",
            "total_ms": round(total_ms, 2),
            "waiting_ms": round(waiting_ms, 2),
            "working_ms": round(working_ms, 2),
            "retrying_ms": round(retrying_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "retries_count": retries_count,
            "retrieval_setting": retrieval_setting,
            "is_degraded": is_degraded,
            "citations_count": len(citations),
            "citations": "; ".join(citations),
            "query": query.replace("\n", " ").strip(),
            "answer_snippet": answer[:150].replace("\n", " ").strip() if answer else "",
        }

        with self.lock:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row)
