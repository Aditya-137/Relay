"""src/standin/mock_service.py

Stand-in model service calibrated directly from empirical Phase 2 measurements on Groq qwen/qwen3.8-27b.
Features:
1. Calibrated Latency Profile:
   - TTFT log-normal distribution: p50 ~115ms, p95 ~388ms
   - Random tail latency spikes: 5% chance of 4.2s - 4.65s cluster contention delay
   - Concurrency degradation: +5ms per active in-flight request
   - Piece-by-piece chunk streaming at ~140 tokens/sec (~7ms per token)
2. Empirical Limit Enforcement:
   - Default: 30 RPM, 7,000 ITPM, 1,000 RPD
   - Configurable for Table 2a scenario: "per-minute limit halved" (15 RPM, 3,500 ITPM)
   - Fast 429 rejection (~45ms) returning retry-after: 2
3. Context-Aware Grounded Response Generation:
   - Parses prompt excerpts, generates valid DrafterOutput JSON with chunk citations [C1], [C2]
   - Refuses if query is unanswerable from provided excerpts
"""

import json
import logging
import math
import random
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StandinMockService:
    """Mock model service replicating Groq free-tier behavior and latency dynamics."""

    def __init__(
        self,
        rpm_limit: int = 30,
        itpm_limit: int = 7000,
        rpd_limit: int = 1000,
        halve_limits: bool = False,
        simulate_tail_spikes: bool = True,
    ):
        self.rpm_limit = rpm_limit // 2 if halve_limits else rpm_limit
        self.itpm_limit = itpm_limit // 2 if halve_limits else itpm_limit
        self.rpd_limit = rpd_limit
        self.simulate_tail_spikes = simulate_tail_spikes

        # Sliding 60s window tracking
        self._request_history = deque()  # (timestamp, 1)
        self._token_history = deque()    # (timestamp, input_tokens)
        self.daily_requests = 0

        # Concurrency tracking
        self.active_in_flight = 0
        self.lock = threading.Lock()

    def _prune_history(self, now: float):
        cutoff = now - 60.0
        while self._request_history and self._request_history[0][0] < cutoff:
            self._request_history.popleft()
        while self._token_history and self._token_history[0][0] < cutoff:
            self._token_history.popleft()

    def get_quota_status(self) -> Dict[str, Any]:
        """Calculates current sliding window utilization against stand-in limits."""
        with self.lock:
            now = time.time()
            self._prune_history(now)
            current_rpm = len(self._request_history)
            current_itpm = sum(t[1] for t in self._token_history)
            rpm_ratio = current_rpm / self.rpm_limit
            itpm_ratio = current_itpm / self.itpm_limit
            max_ratio = max(rpm_ratio, itpm_ratio)
            return {
                "current_rpm": current_rpm,
                "rpm_ratio": round(rpm_ratio, 3),
                "current_itpm": current_itpm,
                "itpm_ratio": round(itpm_ratio, 3),
                "max_ratio": round(max_ratio, 3),
                "is_above_80pct": max_ratio >= 0.80,
                "daily_requests": self.daily_requests,
            }

    def _sample_ttft_ms(self) -> float:
        """Samples TTFT from empirical distribution (log-normal p50 ~115ms, p95 ~388ms)."""
        # 5% chance of multi-tenant tail contention spike (4.2s to 4.65s, observed in Phase 2)
        if self.simulate_tail_spikes and random.random() < 0.05:
            return random.uniform(4200.0, 4650.0)

        # Log-normal distribution: mu = ln(115) ≈ 4.745, sigma ≈ 0.62
        raw_ttft = random.lognormvariate(4.745, 0.62)
        # Add concurrency degradation (+5ms per concurrent request)
        concurrency_penalty = max(0, self.active_in_flight - 1) * 5.0
        return max(85.0, raw_ttft + concurrency_penalty)

    def _synthesize_response(self, user_content: str) -> str:
        """Synthesizes realistic grounded DrafterOutput JSON from prompt excerpts."""
        # Extract query and excerpts
        query_match = re.search(r"QUESTION:\s*(.*?)\s*EXCERPTS:", user_content, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ""

        # Extract available chunk tags [C1], [C2], etc.
        chunks_found = re.findall(r"\[C(\d+)\]\s*Paper:\s*(.*?)\s*\|\s*Section:\s*(.*?)\s*\|\s*Page:\s*(.*?)\nContent:\s*(.*?)(?=\n\n\[C|\Z)", user_content, re.DOTALL)

        # Check for obvious unanswerable queries
        unrelated_keywords = ["sourdough", "bread", "chess", "castling", "ppo", "flashattention"]
        if any(kw in query.lower() for kw in unrelated_keywords) or not chunks_found:
            return json.dumps({
                "is_answerable": False,
                "reason": "The provided excerpts do not contain information related to the question.",
                "answer": "",
            })

        # Synthesize grounded answer using the first 1-2 chunks
        top_chunk = chunks_found[0]
        chunk_idx = top_chunk[0]
        paper_title = top_chunk[1]
        chunk_text_preview = top_chunk[4][:120].replace('"', "'").strip()

        # Produce realistic answer
        if "dimension" in query.lower() or "d_model" in query.lower():
            answer = f"In the Transformer architecture, the model dimensionality is d_model = 512 and the inner head dimension is d_k = 64 [C{chunk_idx}]."
        elif "mimi" in query.lower():
            answer = f"Mimi is a neural audio codec that combines semantic and acoustic information, operating at a 12.5 Hz framerate [C{chunk_idx}]."
        elif "chunking" in query.lower():
            answer = f"Dynamic chunking segments sequences based on learned data-dependent boundary predictions rather than fixed window sizes [C{chunk_idx}]."
        else:
            answer = f"Based on the paper excerpts, {chunk_text_preview}... [C{chunk_idx}]."

        return json.dumps({
            "is_answerable": True,
            "reason": "Excerpts provide direct evidence.",
            "answer": answer,
        })

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        """Simulates Groq streaming chat completion with accurate timing and piece-by-piece yielding."""
        now = time.time()
        with self.lock:
            self._prune_history(now)
            input_text = " ".join([m.get("content", "") for m in messages])
            estimated_input_tokens = int(len(input_text.split()) * 1.3) + 20

            current_rpm = len(self._request_history)
            current_itpm = sum(t[1] for t in self._token_history)

            # Check Limits (RPM and ITPM)
            if current_rpm >= self.rpm_limit or (current_itpm + estimated_input_tokens) > self.itpm_limit:
                # Fast 429 failure (~45ms)
                time.sleep(0.045)
                limit_type = "RPM" if current_rpm >= self.rpm_limit else "ITPM"
                err_msg = f"Rate limit reached for model on {limit_type}. Limit: {self.rpm_limit if limit_type == 'RPM' else self.itpm_limit}"
                logger.warning("StandinMockService 429 (%s): %s", limit_type, err_msg)
                return {
                    "text": "",
                    "ttft_ms": None,
                    "working_ms": 45.0,
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": 0,
                    "status": "rate_limited",
                    "error_message": err_msg,
                    "retry_after_s": 2.0,
                }

            # Check Daily Limit
            if self.daily_requests >= self.rpd_limit:
                time.sleep(0.045)
                return {
                    "text": "",
                    "ttft_ms": None,
                    "working_ms": 45.0,
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": 0,
                    "status": "daily_budget_exhausted",
                    "error_message": "Daily request limit reached (1000/1000 RPD).",
                    "retry_after_s": None,
                }

            # Record request in history
            self._request_history.append((now, 1))
            self._token_history.append((now, estimated_input_tokens))
            self.daily_requests += 1
            self.active_in_flight += 1

        t_start = time.perf_counter()
        try:
            # 1. Simulate Time To First Token (TTFT)
            ttft_ms = self._sample_ttft_ms()
            time.sleep(ttft_ms / 1000.0)

            # 2. Synthesize response text
            user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            full_json_text = self._synthesize_response(user_msg)

            # 3. Simulate piece-by-piece chunk streaming (~7ms per token chunk)
            tokens = re.findall(r"\S+|\s+", full_json_text)
            collected = []
            for token in tokens:
                time.sleep(0.007)  # ~140 tokens/sec
                collected.append(token)

            t_end = time.perf_counter()
            working_ms = (t_end - t_start) * 1000.0
            output_tokens = int(len(full_json_text.split()) * 1.3)

            return {
                "text": full_json_text,
                "ttft_ms": round(ttft_ms, 2),
                "working_ms": round(working_ms, 2),
                "input_tokens": estimated_input_tokens,
                "output_tokens": output_tokens,
                "status": "success",
                "error_message": None,
                "retry_after_s": None,
            }

        finally:
            with self.lock:
                self.active_in_flight = max(0, self.active_in_flight - 1)

