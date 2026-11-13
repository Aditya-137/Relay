"""src/pipeline/orchestrator.py

Main 5-step RAG Pipeline Orchestrator for Vaani:
1. Tool Call: Hybrid Retrieval (FAISS + BM25) across presets 256, 512, 1024
2. Local Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2) or direct hybrid pass
3. Score Gate: Confidence threshold check (< 0.0 -> refuse, 0 Groq tokens)
4. Grounded Drafter: Streaming LLM call (Groq real or Calibrated Stand-in mock)
5. Citation Resolver & Validator: Canonical mapping and hallucination check
Branch/Loop: Retry loop (max_retries=1) if verification fails or unanswerable.
Adaptive Degradation: Swaps to preset_256 + top_k=2 + Caveman-Lite at >80% quota usage.
Rate Limiter & Queue: Two-tier token bucket (per-user fair share + global Groq protection).
Telemetry: Logs per-request metrics with user_id and source to results/results.csv.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from src.pipeline.citation_resolver import CitationResolver
from src.pipeline.models import DrafterOutput, LatencyBreakdown, PipelineResult, TokenUsage
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.reranker import CrossEncoderReranker
from src.standin.mock_service import StandinMockService
from src.utils.groq_client import GroqClientWrapper
from src.utils.metrics_collector import MetricsCollector
from src.utils.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "pipeline.yaml"


class VaaniPipeline:
    """The core multi-step RAG orchestrator for arXiv research paper Q&A."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        standin_client: Optional[StandinMockService] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_cfg = self.config["model"]
        self.retrieval_cfg = self.config["retrieval"]
        self.orch_cfg = self.config["orchestration"]

        # 1. Initialize Hybrid Searchers for all 3 Presets (256, 512, 1024)
        logger.info("Initializing searchers for presets 256, 512, 1024...")
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.searchers = {
            "preset_256": HybridSearcher(index_dir=base_dir / "data" / "indexes" / "preset_256"),
            "preset_512": HybridSearcher(index_dir=base_dir / "data" / "indexes" / "preset_512"),
            "preset_1024": HybridSearcher(index_dir=base_dir / "data" / "indexes" / "preset_1024"),
        }

        # 2. Initialize Local Cross-Encoder
        logger.info("Initializing local Cross-Encoder...")
        self.reranker = CrossEncoderReranker(self.retrieval_cfg["reranker_model"])

        # 3. Initialize Clients (Real Groq & Stand-in Mock)
        logger.info("Initializing Groq and Stand-in clients...")
        self.groq_client = GroqClientWrapper(model_name=self.model_cfg["name"])
        self.standin_client = standin_client or StandinMockService()

        # 4. Initialize Rate Limiter & Telemetry Collector
        giveup_timeout = float(self.orch_cfg.get("timeouts", {}).get("total_max_ms", 10000)) / 1000.0
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(user_giveup_timeout_s=giveup_timeout)
        self.metrics_collector = metrics_collector or MetricsCollector()

    def _build_drafter_prompt(
        self, query: str, chunks: List[Dict[str, Any]], degraded: bool = False
    ) -> List[Dict[str, str]]:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[C{i}] Paper: {chunk['paper_title']} | Section: {chunk['section']} | Page: {chunk['page_start']}\n"
                f"Content: {chunk['text'].strip()}"
            )
        context_str = "\n\n".join(context_parts)

        if degraded:
            system_instruction = (
                "You are Vaani, an expert research assistant. Quota is tight: be ultra-concise and direct.\n"
                "RULES:\n"
                "1. Answer ONLY using evidence from the provided excerpts [C1], [C2]...\n"
                "2. If excerpts lack evidence, set is_answerable=false, reason='Not in excerpts', answer=''.\n"
                "3. Cite every fact using [C1], [C2]. No filler words. Short telegraphic sentences.\n"
                "4. Output STRICT JSON conforming to schema: {\"is_answerable\": bool, \"reason\": str, \"answer\": str}."
            )
        else:
            system_instruction = (
                "You are Vaani, a rigorous research assistant answering engineers' questions about arXiv papers.\n"
                "RULES:\n"
                "1. Answer ONLY using evidence from the provided excerpts [C1], [C2]...\n"
                "2. If the excerpts do not contain sufficient evidence to answer, set is_answerable=false, "
                "provide a clear reason, and set answer to ''.\n"
                "3. You must cite where every claim comes from using chunk tags like [C1], [C2]. "
                "Never cite tags that do not exist.\n"
                "4. Output STRICT JSON conforming to schema: {\"is_answerable\": bool, \"reason\": str, \"answer\": str}."
            )

        user_content = (
            f"QUESTION:\n{query}\n\n"
            f"EXCERPTS:\n{context_str}\n\n"
            f"Produce your JSON evaluation and cited answer."
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ]

    def ask(
        self,
        query: str,
        user_id: str = "default_user",
        scenario: str = "interactive",
        source: str = "real",
        request_id: Optional[str] = None,
        custom_preset: Optional[str] = None,
        custom_top_k: Optional[int] = None,
        use_reranker: bool = True,
    ) -> PipelineResult:
        """Runs the 5-step pipeline with configurable retrieval settings for Table 3 benchmarks."""
        req_id = request_id or str(uuid.uuid4())[:8]
        t_global_start = time.perf_counter()
        query = query.strip()

        # Select model client based on source
        client = self.groq_client if source == "real" else self.standin_client

        # Step 0: Check Adaptive Degradation
        quota_status = client.get_quota_status()
        is_degraded = quota_status["is_above_80pct"]

        chosen_preset = custom_preset or (self.orch_cfg["degraded_preset"] if is_degraded else self.retrieval_cfg["default_preset"])
        top_k = custom_top_k or (self.orch_cfg["degraded_top_k"] if is_degraded else self.retrieval_cfg["top_k"])
        searcher = self.searchers.get(chosen_preset, self.searchers["preset_512"])

        rerank_label = "rerank_yes" if use_reranker else "rerank_no"
        setting_label = f"{chosen_preset}_k{top_k}_{rerank_label}"
        if is_degraded and not custom_preset:
            setting_label += "_degraded"

        candidate_k = self.retrieval_cfg["candidate_k"] if use_reranker else top_k
        rejection_threshold = self.retrieval_cfg["rejection_threshold"]
        max_retries = self.orch_cfg["max_retries"]

        # ------------------------------------------------------------------
        # Step 0.5: Two-Tier Rate Limiter Admission & Queueing
        # ------------------------------------------------------------------
        estimated_needed = 700 if is_degraded else (top_k * 400 + 100)
        admitted, queue_waiting_ms, admit_status = self.rate_limiter.acquire(
            user_id=user_id, estimated_input_tokens=estimated_needed
        )

        if not admitted:
            t_global_end = time.perf_counter()
            total_latency_ms = (t_global_end - t_global_start) * 1000.0

            if admit_status == "daily_budget_exhausted":
                msg = (
                    "The daily query allocation for this service has been exhausted. "
                    "Please try again later when the 24-hour rolling quota resets."
                )
                status_code = "daily_budget_exhausted"
            else:
                msg = "Request timed out in queue before processing could begin due to high system load."
                status_code = "timed_out"

            res = PipelineResult(
                query=query,
                answer=msg,
                status=status_code,
                is_answerable=False,
                latency=LatencyBreakdown(
                    waiting_ms=queue_waiting_ms,
                    working_ms=0.0,
                    retrying_ms=0.0,
                    ttft_ms=None,
                    total_ms=round(total_latency_ms, 2),
                ),
                token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                retry_count=0,
                degraded=is_degraded,
            )
            self.metrics_collector.log_request(
                request_id=req_id,
                query=query,
                user_id=user_id,
                status=status_code,
                scenario=scenario,
                source=source,
                retrieval_setting=setting_label,
                total_ms=total_latency_ms,
                waiting_ms=queue_waiting_ms,
                is_degraded=is_degraded,
                answer=msg,
            )
            return res

        # Admitted to pipeline
        retry_count = 0
        total_waiting_ms = queue_waiting_ms
        total_retrying_ms = 0.0
        accumulated_input_tokens = 0
        accumulated_output_tokens = 0
        first_ttft_ms: Optional[float] = None

        current_query = query

        while retry_count <= max_retries:
            t_attempt_start = time.perf_counter()

            # ------------------------------------------------------------------
            # Step 1: Hybrid Retrieval Tool
            # ------------------------------------------------------------------
            candidates = searcher.search(
                current_query, top_k=candidate_k, alpha=self.retrieval_cfg["alpha"]
            )

            # ------------------------------------------------------------------
            # Step 2: Reranking (Cross-Encoder or Direct Hybrid Pass)
            # ------------------------------------------------------------------
            if use_reranker:
                reranked_chunks, top1_score = self.reranker.rerank(
                    current_query, candidates, top_k=top_k
                )
            else:
                reranked_chunks = candidates[:top_k]
                top1_score = reranked_chunks[0]["score"] if reranked_chunks else -999.0
                # Map score to relative threshold scale if not using cross-encoder
                for c in reranked_chunks:
                    c["rerank_score"] = c["score"]

            # ------------------------------------------------------------------
            # Step 3: Confidence Score Gate (Refuse Out-of-Domain)
            # ------------------------------------------------------------------
            gate_cutoff = rejection_threshold if use_reranker else 0.15
            if top1_score < gate_cutoff:
                t_global_end = time.perf_counter()
                total_latency_ms = (t_global_end - t_global_start) * 1000.0
                refusal_msg = (
                    "I cannot answer this question because it is not covered in the three research papers "
                    "(Attention Is All You Need, Moshi, Dynamic Chunking)."
                )
                res = PipelineResult(
                    query=query,
                    answer=refusal_msg,
                    status="refused_score_gate",
                    is_answerable=False,
                    citations=[],
                    retrieved_chunks=reranked_chunks,
                    latency=LatencyBreakdown(
                        waiting_ms=round(total_waiting_ms, 2),
                        working_ms=round(total_latency_ms - total_waiting_ms, 2),
                        retrying_ms=0.0,
                        ttft_ms=None,
                        total_ms=round(total_latency_ms, 2),
                    ),
                    token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    retry_count=retry_count,
                    degraded=is_degraded,
                )
                self.metrics_collector.log_request(
                    request_id=req_id,
                    query=query,
                    user_id=user_id,
                    status="refused_score_gate",
                    scenario=scenario,
                    source=source,
                    retrieval_setting=setting_label,
                    total_ms=total_latency_ms,
                    waiting_ms=total_waiting_ms,
                    working_ms=total_latency_ms - total_waiting_ms,
                    is_degraded=is_degraded,
                    answer=refusal_msg,
                )
                return res

            # ------------------------------------------------------------------
            # Step 4: Grounded Drafter (Groq Real or Stand-in Mock)
            # ------------------------------------------------------------------
            messages = self._build_drafter_prompt(query, reranked_chunks, degraded=is_degraded)
            llm_res = client.chat_stream(
                messages=messages,
                temperature=self.model_cfg["temperature"],
                max_tokens=self.model_cfg["max_output_tokens"],
                json_mode=self.model_cfg["json_mode"],
            )

            if first_ttft_ms is None and llm_res["ttft_ms"] is not None:
                first_ttft_ms = llm_res["ttft_ms"]

            accumulated_input_tokens += llm_res["input_tokens"]
            accumulated_output_tokens += llm_res["output_tokens"]

            # Handle 429 Rate Limit
            if llm_res["status"] == "rate_limited":
                if retry_count < max_retries:
                    retry_count += 1
                    sleep_s = llm_res.get("retry_after_s") or 2.0
                    total_waiting_ms += sleep_s * 1000.0
                    time.sleep(sleep_s)
                    t_attempt_end = time.perf_counter()
                    total_retrying_ms += (t_attempt_end - t_attempt_start) * 1000.0
                    continue
                else:
                    t_global_end = time.perf_counter()
                    total_latency_ms = (t_global_end - t_global_start) * 1000.0
                    err_msg = "Request could not be completed because model rate limits were exhausted."
                    res = PipelineResult(
                        query=query,
                        answer=err_msg,
                        status="rate_limited",
                        is_answerable=False,
                        latency=LatencyBreakdown(
                            waiting_ms=round(total_waiting_ms, 2),
                            working_ms=round(total_latency_ms - total_waiting_ms - total_retrying_ms, 2),
                            retrying_ms=round(total_retrying_ms, 2),
                            ttft_ms=first_ttft_ms,
                            total_ms=round(total_latency_ms, 2),
                        ),
                        token_usage=TokenUsage(
                            input_tokens=accumulated_input_tokens,
                            output_tokens=accumulated_output_tokens,
                            total_tokens=accumulated_input_tokens + accumulated_output_tokens,
                        ),
                        retry_count=retry_count,
                        degraded=is_degraded,
                    )
                    self.metrics_collector.log_request(
                        request_id=req_id,
                        query=query,
                        user_id=user_id,
                        status="rate_limited",
                        scenario=scenario,
                        source=source,
                        retrieval_setting=setting_label,
                        ttft_ms=first_ttft_ms,
                        total_ms=total_latency_ms,
                        waiting_ms=total_waiting_ms,
                        working_ms=total_latency_ms - total_waiting_ms - total_retrying_ms,
                        retrying_ms=total_retrying_ms,
                        input_tokens=accumulated_input_tokens,
                        output_tokens=accumulated_output_tokens,
                        retries_count=retry_count,
                        is_degraded=is_degraded,
                        answer=err_msg,
                    )
                    return res

            # Parse JSON output from Drafter
            try:
                parsed_json = json.loads(llm_res["text"])
                drafter_out = DrafterOutput.model_validate(parsed_json)
            except (json.JSONDecodeError, ValidationError) as err:
                logger.warning("Drafter JSON validation error: %s", err)
                if retry_count < max_retries:
                    retry_count += 1
                    t_attempt_end = time.perf_counter()
                    total_retrying_ms += (t_attempt_end - t_attempt_start) * 1000.0
                    continue
                else:
                    t_global_end = time.perf_counter()
                    total_latency_ms = (t_global_end - t_global_start) * 1000.0
                    err_msg = "Error generating structured response from model."
                    res = PipelineResult(
                        query=query,
                        answer=err_msg,
                        status="error",
                        is_answerable=False,
                        latency=LatencyBreakdown(
                            waiting_ms=round(total_waiting_ms, 2),
                            working_ms=round(total_latency_ms - total_waiting_ms - total_retrying_ms, 2),
                            retrying_ms=round(total_retrying_ms, 2),
                            ttft_ms=first_ttft_ms,
                            total_ms=round(total_latency_ms, 2),
                        ),
                        token_usage=TokenUsage(
                            input_tokens=accumulated_input_tokens,
                            output_tokens=accumulated_output_tokens,
                            total_tokens=accumulated_input_tokens + accumulated_output_tokens,
                        ),
                        retry_count=retry_count,
                        degraded=is_degraded,
                    )
                    self.metrics_collector.log_request(
                        request_id=req_id,
                        query=query,
                        user_id=user_id,
                        status="error",
                        scenario=scenario,
                        source=source,
                        retrieval_setting=setting_label,
                        ttft_ms=first_ttft_ms,
                        total_ms=total_latency_ms,
                        waiting_ms=total_waiting_ms,
                        working_ms=total_latency_ms - total_waiting_ms - total_retrying_ms,
                        retrying_ms=total_retrying_ms,
                        input_tokens=accumulated_input_tokens,
                        output_tokens=accumulated_output_tokens,
                        retries_count=retry_count,
                        is_degraded=is_degraded,
                        answer=err_msg,
                    )
                    return res

            # ------------------------------------------------------------------
            # Step 5: Citation Resolver & Grounding Verification
            # ------------------------------------------------------------------
            if not drafter_out.is_answerable:
                if retry_count < max_retries:
                    retry_count += 1
                    candidate_k += 5
                    t_attempt_end = time.perf_counter()
                    total_retrying_ms += (t_attempt_end - t_attempt_start) * 1000.0
                    continue
                else:
                    t_global_end = time.perf_counter()
                    total_latency_ms = (t_global_end - t_global_start) * 1000.0
                    ans_str = f"The papers do not contain sufficient information to answer this question. ({drafter_out.reason})"
                    res = PipelineResult(
                        query=query,
                        answer=ans_str,
                        status="refused_unanswerable",
                        is_answerable=False,
                        retrieved_chunks=reranked_chunks,
                        latency=LatencyBreakdown(
                            waiting_ms=round(total_waiting_ms, 2),
                            working_ms=round(total_latency_ms - total_waiting_ms - total_retrying_ms, 2),
                            retrying_ms=round(total_retrying_ms, 2),
                            ttft_ms=first_ttft_ms,
                            total_ms=round(total_latency_ms, 2),
                        ),
                        token_usage=TokenUsage(
                            input_tokens=accumulated_input_tokens,
                            output_tokens=accumulated_output_tokens,
                            total_tokens=accumulated_input_tokens + accumulated_output_tokens,
                        ),
                        retry_count=retry_count,
                        degraded=is_degraded,
                    )
                    self.metrics_collector.log_request(
                        request_id=req_id,
                        query=query,
                        user_id=user_id,
                        status="refused_unanswerable",
                        scenario=scenario,
                        source=source,
                        retrieval_setting=setting_label,
                        ttft_ms=first_ttft_ms,
                        total_ms=total_latency_ms,
                        waiting_ms=total_waiting_ms,
                        working_ms=total_latency_ms - total_waiting_ms - total_retrying_ms,
                        retrying_ms=total_retrying_ms,
                        input_tokens=accumulated_input_tokens,
                        output_tokens=accumulated_output_tokens,
                        retries_count=retry_count,
                        is_degraded=is_degraded,
                        answer=ans_str,
                    )
                    return res

            # Resolve citations and check grounding
            resolved_answer, citations, is_valid, msg = CitationResolver.resolve(
                drafter_out.answer, reranked_chunks
            )

            if not is_valid:
                logger.warning("Citation validation failure: %s", msg)
                if retry_count < max_retries:
                    retry_count += 1
                    t_attempt_end = time.perf_counter()
                    total_retrying_ms += (t_attempt_end - t_attempt_start) * 1000.0
                    continue
                else:
                    t_global_end = time.perf_counter()
                    total_latency_ms = (t_global_end - t_global_start) * 1000.0
                    res = PipelineResult(
                        query=query,
                        answer=resolved_answer,
                        status="failed_validation",
                        is_answerable=True,
                        citations=citations,
                        retrieved_chunks=reranked_chunks,
                        latency=LatencyBreakdown(
                            waiting_ms=round(total_waiting_ms, 2),
                            working_ms=round(total_latency_ms - total_waiting_ms - total_retrying_ms, 2),
                            retrying_ms=round(total_retrying_ms, 2),
                            ttft_ms=first_ttft_ms,
                            total_ms=round(total_latency_ms, 2),
                        ),
                        token_usage=TokenUsage(
                            input_tokens=accumulated_input_tokens,
                            output_tokens=accumulated_output_tokens,
                            total_tokens=accumulated_input_tokens + accumulated_output_tokens,
                        ),
                        retry_count=retry_count,
                        degraded=is_degraded,
                    )
                    self.metrics_collector.log_request(
                        request_id=req_id,
                        query=query,
                        user_id=user_id,
                        status="failed_validation",
                        scenario=scenario,
                        source=source,
                        retrieval_setting=setting_label,
                        ttft_ms=first_ttft_ms,
                        total_ms=total_latency_ms,
                        waiting_ms=total_waiting_ms,
                        working_ms=total_latency_ms - total_waiting_ms - total_retrying_ms,
                        retrying_ms=total_retrying_ms,
                        input_tokens=accumulated_input_tokens,
                        output_tokens=accumulated_output_tokens,
                        retries_count=retry_count,
                        is_degraded=is_degraded,
                        citations=citations,
                        answer=resolved_answer,
                    )
                    return res

            # Success!
            t_global_end = time.perf_counter()
            total_latency_ms = (t_global_end - t_global_start) * 1000.0
            res = PipelineResult(
                query=query,
                answer=resolved_answer,
                status="success",
                is_answerable=True,
                citations=citations,
                retrieved_chunks=reranked_chunks,
                latency=LatencyBreakdown(
                    waiting_ms=round(total_waiting_ms, 2),
                    working_ms=round(total_latency_ms - total_waiting_ms - total_retrying_ms, 2),
                    retrying_ms=round(total_retrying_ms, 2),
                    ttft_ms=first_ttft_ms,
                    total_ms=round(total_latency_ms, 2),
                ),
                token_usage=TokenUsage(
                    input_tokens=accumulated_input_tokens,
                    output_tokens=accumulated_output_tokens,
                    total_tokens=accumulated_input_tokens + accumulated_output_tokens,
                ),
                retry_count=retry_count,
                degraded=is_degraded,
            )
            self.metrics_collector.log_request(
                request_id=req_id,
                query=query,
                user_id=user_id,
                status="success",
                scenario=scenario,
                source=source,
                retrieval_setting=setting_label,
                ttft_ms=first_ttft_ms,
                total_ms=total_latency_ms,
                waiting_ms=total_waiting_ms,
                working_ms=total_latency_ms - total_waiting_ms - total_retrying_ms,
                retrying_ms=total_retrying_ms,
                input_tokens=accumulated_input_tokens,
                output_tokens=accumulated_output_tokens,
                retries_count=retry_count,
                is_degraded=is_degraded,
                citations=citations,
                answer=resolved_answer,
            )
            return res

        # Fallback if loop finishes unexpectedly
        t_global_end = time.perf_counter()
        total_latency_ms = (t_global_end - t_global_start) * 1000.0
        err_msg = "Pipeline could not complete request within retry limits."
        res = PipelineResult(
            query=query,
            answer=err_msg,
            status="error",
            is_answerable=False,
            latency=LatencyBreakdown(
                waiting_ms=round(total_waiting_ms, 2),
                working_ms=round(total_latency_ms - total_waiting_ms - total_retrying_ms, 2),
                retrying_ms=round(total_retrying_ms, 2),
                ttft_ms=first_ttft_ms,
                total_ms=round(total_latency_ms, 2),
            ),
            token_usage=TokenUsage(
                input_tokens=accumulated_input_tokens,
                output_tokens=accumulated_output_tokens,
                total_tokens=accumulated_input_tokens + accumulated_output_tokens,
            ),
            retry_count=retry_count,
            degraded=is_degraded,
        )
        self.metrics_collector.log_request(
            request_id=req_id,
            query=query,
            user_id=user_id,
            status="error",
            scenario=scenario,
            source=source,
            retrieval_setting=setting_label,
            ttft_ms=first_ttft_ms,
            total_ms=total_latency_ms,
            waiting_ms=total_waiting_ms,
            working_ms=total_latency_ms - total_waiting_ms - total_retrying_ms,
            retrying_ms=total_retrying_ms,
            input_tokens=accumulated_input_tokens,
            output_tokens=accumulated_output_tokens,
            retries_count=retry_count,
            is_degraded=is_degraded,
            answer=err_msg,
        )
        return res

    run = ask


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run Vaani RAG Pipeline on arXiv papers")
    parser.add_argument("query", nargs="?", type=str, default="What is multi-head attention?", help="User query")
    parser.add_argument("--source", choices=["real", "standin"], default="standin", help="Model backend ('real' for Groq, 'standin' for mock)")
    parser.add_argument("--preset", choices=["preset_256", "preset_512", "preset_1024"], default="preset_512", help="Chunk preset")
    parser.add_argument("--no-rerank", action="store_true", help="Disable local Cross-Encoder reranking")

    args = parser.parse_args()

    pipeline = VaaniPipeline()
    result = pipeline.run(
        query=args.query,
        source=args.source,
        custom_preset=args.preset,
        use_reranker=not args.no_rerank,
    )

    print("\n" + "=" * 60)
    print(f"QUERY:    {result.query}")
    print(f"STATUS:   {result.status} (Answerable: {result.is_answerable})")
    print(f"LATENCY:  TTFT: {result.latency.ttft_ms}ms | Total: {result.latency.total_ms}ms")
    print(f"CITATIONS: {result.citations}")
    print(f"ANSWER:\n{result.answer}")
    print("=" * 60 + "\n")

