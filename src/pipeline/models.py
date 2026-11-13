"""src/pipeline/models.py

Pydantic data schemas for pipeline inputs, outputs, telemetry, and Drafter JSON.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DrafterOutput(BaseModel):
    """Structured JSON output contract for Groq Drafter."""
    is_answerable: bool = Field(
        ...,
        description="True if the provided excerpts contain sufficient evidence to answer; False otherwise.",
    )
    reason: str = Field(
        ...,
        description="Brief 1-sentence justification for whether the excerpts answer the question.",
    )
    answer: str = Field(
        ...,
        description="The cited answer citing [C1], [C2], etc. If is_answerable is False, this must be an empty string.",
    )


class LatencyBreakdown(BaseModel):
    """Detailed latency telemetry per request, required by Assignment Section 4."""
    waiting_ms: float = 0.0     # Time spent waiting before execution (rate-limiter / queue)
    working_ms: float = 0.0     # Time spent executing (retrieval + rerank + model inference)
    retrying_ms: float = 0.0    # Time spent on retry loops
    ttft_ms: Optional[float] = None  # Time To First Token for model streaming
    total_ms: float = 0.0       # Total end-to-end elapsed time


class TokenUsage(BaseModel):
    """Token consumption telemetry."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class PipelineResult(BaseModel):
    """Final result emitted by the Vaani RAG pipeline."""
    query: str
    answer: str
    status: str                 # success | refused_score_gate | refused_unanswerable | failed_validation | rate_limited | error
    is_answerable: bool = True
    citations: List[str] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    retry_count: int = 0
    degraded: bool = False      # True if adaptive degradation (RT + Caveman) was triggered

