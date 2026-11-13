"""src/utils/groq_client.py

Groq API client wrapper with:
1. Streaming support for exact TTFT measurement.
2. In-memory sliding-window token & request quota tracking.
3. Adaptive degradation signal (detects when quota usage > 80%).
4. Structured latency breakdown (waiting, working, retrying).
"""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


class GroqClientWrapper:
    """Production Groq client wrapper with streaming telemetry and rate tracking."""

    # Empirical limits from Phase 2
    LIMIT_RPM = 30
    LIMIT_ITPM = 7000
    LIMIT_RPD = 1000

    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set in .env")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

        # Rolling 60s window tracking: deques of (timestamp, value)
        self._request_history = deque()  # (timestamp, 1)
        self._token_history = deque()    # (timestamp, input_tokens)
        self._daily_requests = 0

    def _prune_history(self, now: float):
        """Remove entries older than 60 seconds from sliding windows."""
        cutoff = now - 60.0
        while self._request_history and self._request_history[0][0] < cutoff:
            self._request_history.popleft()
        while self._token_history and self._token_history[0][0] < cutoff:
            self._token_history.popleft()

    def get_quota_status(self) -> Dict[str, Any]:
        """Calculates current sliding window utilization against Groq limits."""
        now = time.time()
        self._prune_history(now)

        current_rpm = len(self._request_history)
        current_itpm = sum(t[1] for t in self._token_history)

        rpm_ratio = current_rpm / self.LIMIT_RPM
        itpm_ratio = current_itpm / self.LIMIT_ITPM
        max_ratio = max(rpm_ratio, itpm_ratio)

        return {
            "current_rpm": current_rpm,
            "rpm_ratio": round(rpm_ratio, 3),
            "current_itpm": current_itpm,
            "itpm_ratio": round(itpm_ratio, 3),
            "max_ratio": round(max_ratio, 3),
            "is_above_80pct": max_ratio >= 0.80,
            "daily_requests": self._daily_requests,
        }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        """Executes a streaming chat completion, recording exact TTFT and latency.

        Returns:
            Dict containing:
            - text: Full raw accumulated output string
            - ttft_ms: Time To First Token in ms
            - working_ms: Duration of request execution in ms
            - input_tokens: Estimated input tokens
            - output_tokens: Estimated output tokens
            - status: "success" | "rate_limited" | "error"
            - error_message: str or None
            - retry_after_s: float or None
        """
        now = time.time()
        self._prune_history(now)

        # Approximate input tokens (~1.3 tokens per word)
        input_text = " ".join([m.get("content", "") for m in messages])
        estimated_input_tokens = int(len(input_text.split()) * 1.3) + 20

        # Record attempt in history
        self._request_history.append((now, 1))
        self._token_history.append((now, estimated_input_tokens))
        self._daily_requests += 1

        response_format = {"type": "json_object"} if json_mode else None

        t_start = time.perf_counter()
        ttft_ms: Optional[float] = None
        collected_chunks = []

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    collected_chunks.append(delta)

            t_end = time.perf_counter()
            working_ms = (t_end - t_start) * 1000.0
            full_text = "".join(collected_chunks)
            estimated_output_tokens = int(len(full_text.split()) * 1.3)

            return {
                "text": full_text,
                "ttft_ms": round(ttft_ms or working_ms, 2),
                "working_ms": round(working_ms, 2),
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "status": "success",
                "error_message": None,
                "retry_after_s": None,
            }

        except RateLimitError as e:
            t_end = time.perf_counter()
            working_ms = (t_end - t_start) * 1000.0
            # Groq 429 often provides retry-after in headers or body
            retry_after = 2.0
            try:
                if hasattr(e, "response") and e.response:
                    retry_after = float(e.response.headers.get("retry-after", 2.0))
            except Exception:
                pass

            logger.warning("Groq RateLimitError (429): %s", e)
            return {
                "text": "",
                "ttft_ms": None,
                "working_ms": round(working_ms, 2),
                "input_tokens": estimated_input_tokens,
                "output_tokens": 0,
                "status": "rate_limited",
                "error_message": str(e),
                "retry_after_s": retry_after,
            }

        except Exception as e:
            t_end = time.perf_counter()
            working_ms = (t_end - t_start) * 1000.0
            logger.error("Groq API error: %s", e)
            return {
                "text": "",
                "ttft_ms": None,
                "working_ms": round(working_ms, 2),
                "input_tokens": estimated_input_tokens,
                "output_tokens": 0,
                "status": "error",
                "error_message": str(e),
                "retry_after_s": None,
            }

