"""src/utils/rate_limiter.py

Thread-safe dual-tier Token-Bucket Rate Limiter & Concurrency Queue.
Enforces:
1. Global Provider Limits:
   - 30 RPM (Groq limit, refill: 0.5 req/s)
   - 7,000 ITPM (Groq limit, refill: 116.67 tok/s)
   - 1,000 RPD (Groq daily limit)
2. Per-User Fair-Share Limits:
   - Max 2 req/min per user (calibrated because whole pipeline capacity is ~3-4 req/min due to 7k ITPM)
   - Max 2,500 input tokens/min per user
3. Queueing & User Give-Up:
   - Drops requests waiting > timeout (default 10s) with status "timed_out"
   - Tracks peak queue length for Table 2b.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class UserTokenBucket:
    """Individual fair-share bucket per engineer/user."""

    def __init__(self, rpm_limit: float = 2.0, itpm_limit: float = 2500.0):
        self.rpm_limit = rpm_limit
        self.itpm_limit = itpm_limit
        self.req_tokens = float(rpm_limit)
        self.input_tokens = float(itpm_limit)
        self.rpm_refill_rate = rpm_limit / 60.0
        self.itpm_refill_rate = itpm_limit / 60.0
        self.last_refill = time.perf_counter()

    def refill(self, now: float):
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.req_tokens = min(self.rpm_limit, self.req_tokens + elapsed * self.rpm_refill_rate)
            self.input_tokens = min(self.itpm_limit, self.input_tokens + elapsed * self.itpm_refill_rate)
            self.last_refill = now


class TokenBucketRateLimiter:
    """Two-tier token bucket: per-user fair sharing + global Groq protection."""

    def __init__(
        self,
        rpm_limit: int = 30,
        itpm_limit: int = 7000,
        rpd_limit: int = 1000,
        user_rpm_limit: float = 2.0,
        user_itpm_limit: float = 2500.0,
        user_giveup_timeout_s: float = 10.0,
    ):
        self.global_rpm_limit = rpm_limit
        self.global_itpm_limit = itpm_limit
        self.global_rpd_limit = rpd_limit
        self.user_rpm_limit = user_rpm_limit
        self.user_itpm_limit = user_itpm_limit
        self.user_giveup_timeout_s = user_giveup_timeout_s

        # Global refill rates
        self.global_rpm_refill_rate = rpm_limit / 60.0
        self.global_itpm_refill_rate = itpm_limit / 60.0

        # Global buckets
        self.global_req_tokens = float(rpm_limit)
        self.global_input_tokens = float(itpm_limit)
        self.daily_requests_used = 0
        self.last_global_refill = time.perf_counter()

        # Per-user buckets
        self.user_buckets: Dict[str, UserTokenBucket] = {}

        # Concurrency & queue
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.current_queue_length = 0
        self.peak_queue_length = 0

    def _get_user_bucket(self, user_id: str) -> UserTokenBucket:
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = UserTokenBucket(
                rpm_limit=self.user_rpm_limit, itpm_limit=self.user_itpm_limit
            )
        return self.user_buckets[user_id]

    def _refill_global(self, now: float):
        elapsed = now - self.last_global_refill
        if elapsed > 0:
            self.global_req_tokens = min(
                float(self.global_rpm_limit),
                self.global_req_tokens + elapsed * self.global_rpm_refill_rate,
            )
            self.global_input_tokens = min(
                float(self.global_itpm_limit),
                self.global_input_tokens + elapsed * self.global_itpm_refill_rate,
            )
            self.last_global_refill = now

    def acquire(
        self,
        user_id: str = "default_user",
        estimated_input_tokens: int = 1200,
        timeout_s: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """Attempts to acquire rate-limit admission for a specific user."""
        max_wait = timeout_s if timeout_s is not None else self.user_giveup_timeout_s
        t_entry = time.perf_counter()

        with self.lock:
            # Check Daily Limit
            if self.daily_requests_used >= self.global_rpd_limit:
                logger.warning("Daily quota reached (%d/%d RPD)", self.daily_requests_used, self.global_rpd_limit)
                return False, 0.0, "daily_budget_exhausted"

            user_b = self._get_user_bucket(user_id)
            self.current_queue_length += 1
            if self.current_queue_length > self.peak_queue_length:
                self.peak_queue_length = self.current_queue_length

            try:
                while True:
                    now = time.perf_counter()
                    self._refill_global(now)
                    user_b.refill(now)
                    elapsed_wait = now - t_entry

                    # Check capacities
                    user_ok = user_b.req_tokens >= 1.0 and user_b.input_tokens >= estimated_input_tokens
                    global_ok = self.global_req_tokens >= 1.0 and self.global_input_tokens >= estimated_input_tokens

                    if user_ok and global_ok:
                        user_b.req_tokens -= 1.0
                        user_b.input_tokens -= estimated_input_tokens
                        self.global_req_tokens -= 1.0
                        self.global_input_tokens -= estimated_input_tokens
                        self.daily_requests_used += 1
                        waiting_ms = round(elapsed_wait * 1000.0, 2)
                        return True, waiting_ms, "admitted"

                    # Check give-up timeout
                    remaining_timeout = max_wait - elapsed_wait
                    if remaining_timeout <= 0:
                        waiting_ms = round(elapsed_wait * 1000.0, 2)
                        reason = "user_throttled" if not user_ok else "global_overload"
                        logger.warning("Request from %s timed out (%s) after waiting %.2f ms", user_id, reason, waiting_ms)
                        return False, waiting_ms, "timed_out"

                    time_to_sleep = min(remaining_timeout, 0.1)
                    self.condition.wait(timeout=time_to_sleep)

            finally:
                self.current_queue_length -= 1
                self.condition.notify_all()

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            now = time.perf_counter()
            self._refill_global(now)
            return {
                "available_global_req_tokens": round(self.global_req_tokens, 2),
                "available_global_input_tokens": round(self.global_input_tokens, 1),
                "daily_requests_used": self.daily_requests_used,
                "current_queue_length": self.current_queue_length,
                "peak_queue_length": self.peak_queue_length,
                "active_users_tracked": len(self.user_buckets),
            }

    def reset_stats(self):
        with self.lock:
            self.peak_queue_length = self.current_queue_length
