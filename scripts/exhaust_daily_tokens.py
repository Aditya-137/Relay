"""scripts/exhaust_daily_tokens.py

Script to systematically exhaust all tokens from Groq (free tier qwen/qwen3.8-27b)
until the daily token limit (TPD) or daily request limit (RPD) is hit.
Logs exact headers, error payloads, recovery timings, and resets.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import groq

# Load environment
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
    sys.exit(1)

client = groq.Groq(api_key=api_key)
model_name = "qwen/qwen3.8-27b"

# We use ~6,200 input tokens to maximize token burn while staying safely below the 7,000 ITPM limit
base_sentence = "Artificial intelligence and hierarchical sequence modeling require robust attention mechanisms. "
repeat_count = 550  # ~6,000 to 6,400 tokens
prompt_text = base_sentence * repeat_count

print("=" * 70)
print(f"Starting Groq Token Exhaustion on {model_name}")
print(f"Timestamp: {datetime.now().isoformat()}")
print("=" * 70)

total_tokens_burned = 0
request_idx = 0
daily_limit_hit = False
exhaustion_result = {}

while not daily_limit_hit:
    request_idx += 1
    t0 = time.time()
    try:
        raw_resp = client.chat.completions.with_raw_response.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=60,
            temperature=0.0,
        )
        t_req = time.time() - t0
        headers = dict(raw_resp.headers)
        usage = raw_resp.parse().usage

        prompt_tok = usage.prompt_tokens
        comp_tok = usage.completion_tokens
        req_total_tok = usage.total_tokens
        total_tokens_burned += req_total_tok

        rem_req = headers.get("x-ratelimit-remaining-requests", "unknown")
        rem_tok = headers.get("x-ratelimit-remaining-tokens", "unknown")
        reset_req = headers.get("x-ratelimit-reset-requests", "unknown")
        reset_tok = headers.get("x-ratelimit-reset-tokens", "0s")

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Req #{request_idx:02d} OK ({t_req:.2f}s) | "
            f"Burned: {req_total_tok} tok (Total: {total_tokens_burned}) | "
            f"Rem Req: {rem_req} | Rem Tok: {rem_tok} | Reset Tok: {reset_tok} | Reset Req: {reset_req}"
        )

        # Parse reset_tok to sleep
        sleep_s = 55.0  # safe default for 6k tokens at 116.7 tok/s
        if reset_tok:
            try:
                # format could be "52.34s" or "123ms"
                if "ms" in reset_tok:
                    sleep_s = float(reset_tok.replace("ms", "")) / 1000.0 + 1.0
                elif "s" in reset_tok:
                    sleep_s = float(reset_tok.replace("s", "")) + 1.0
            except Exception:
                sleep_s = 55.0

        # Ensure we sleep at least enough to refill the burned tokens
        min_refill_wait = (prompt_tok / 116.7) + 2.0
        final_sleep = max(sleep_s, min_refill_wait)
        print(f"  -> Sleeping {final_sleep:.1f}s to refill per-minute bucket...")
        time.sleep(final_sleep)

    except groq.APIStatusError as e:
        status_code = e.status_code
        err_msg = e.message
        headers = dict(e.response.headers)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] HTTP {status_code}: {err_msg}")

        # Check if this is the daily limit!
        is_daily = (
            "day" in err_msg.lower()
            or "tpd" in err_msg.lower()
            or "rpd" in err_msg.lower()
            or "daily" in err_msg.lower()
        )

        if is_daily:
            print("\n" + "!" * 70)
            print("🎯 DAILY LIMIT EXHAUSTED!")
            print(f"Status Code: {status_code}")
            print(f"Error Message: {err_msg}")
            print("Headers:")
            for k, v in headers.items():
                if "ratelimit" in k.lower() or "retry" in k.lower():
                    print(f"  {k}: {v}")
            print("!" * 70 + "\n")

            exhaustion_result = {
                "timestamp": datetime.now().isoformat(),
                "status_code": status_code,
                "error_message": err_msg,
                "total_tokens_burned_this_session": total_tokens_burned,
                "requests_sent_this_session": request_idx,
                "headers": headers,
            }
            daily_limit_hit = True
            break

        elif status_code == 429:
            # Per-minute rate limit hit (e.g. ITPM or RPM)
            retry_after = headers.get("retry-after")
            try:
                wait_time = float(retry_after) + 1.0 if retry_after else 30.0
            except ValueError:
                wait_time = 30.0
            print(f"  -> Per-minute limit hit. Retry-after: {retry_after}s. Sleeping {wait_time:.1f}s...")
            time.sleep(wait_time)
        else:
            print(f"  -> Unexpected error: {e}. Sleeping 10s...")
            time.sleep(10.0)

    except Exception as ex:
        print(f"Unexpected exception: {type(ex)} {ex}. Sleeping 10s...")
        time.sleep(10.0)

# Save exhaustion proof to results
output_file = Path(__file__).resolve().parent.parent / "results" / "daily_exhaustion_proof.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(exhaustion_result, f, indent=2)

print(f"Saved daily exhaustion proof to {output_file}")

