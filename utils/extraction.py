"""
extraction.py — IFRS 16 contract extraction pipeline

Architecture
------------
All tasks in llmcalls.json are independent of one another (each reads the same
contract text and writes to a separate output key), so they run in parallel via
a ThreadPoolExecutor.  Each task is wrapped with retry logic so a transient API
failure doesn't abort the whole run.

Flow:
  contract.txt ──┐
                 ├──► [ThreadPoolExecutor]
  llmcalls.json ─┘         │
                            ├─ extraction(leaseJudgment)  ──► parse_response()
                            ├─ extraction(leaseTerms)     ──► parse_response()
                            ├─ extraction(paymentTerms)   ──► parse_response()
                            ├─ extraction(costSeparation) ──► parse_response()
                            └─ extraction(discountRate)   ──► parse_response()
                                        │
                                        ▼
                              results dict  ──► results.json
                              (task_name → parsed payload | error envelope)
"""

import json
import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from flask.cli import load_dotenv

from llmapi import LLMClient, extraction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw: Dict[str, Any], task_name: str) -> Any:
    """
    Unpack the API envelope and return only the structured payload.

    Handles two envelope shapes:

    Azure OpenAI / OpenAI:
        { "choices": [{ "message": { "content": "<json string>" } }] }

    Anthropic /v1/messages:
        { "content": [{ "type": "text", "text": "<json string>" }] }

    Returns the parsed inner object, or an error dict on failure.
    """
    if not raw:
        return {"error": "empty_response", "task": task_name}

    try:
        # Azure OpenAI / OpenAI envelope
        if "choices" in raw:
            text = raw["choices"][0]["message"]["content"]
        # Anthropic envelope
        elif "content" in raw:
            text = raw["content"][0]["text"]
        else:
            return {"error": "unrecognised_envelope", "raw": raw, "task": task_name}

        return json.loads(text)

    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        log.warning("[%s] parse_response failed: %s", task_name, exc)
        return {"error": str(exc), "raw_text": raw, "task": task_name}


# ---------------------------------------------------------------------------
# Single-task runner with retry
# ---------------------------------------------------------------------------

def run_task(
    client: LLMClient,
    task_entry: Dict[str, Any],
    contract_text: str,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> Dict[str, Any]:
    """
    Run one extraction task.  Retries on transient errors with exponential
    backoff.  Returns a result envelope:

        {
            "task":    "Extract leaseJudgment Info",
            "status":  "ok" | "error",
            "payload": <parsed JSON> | <error dict>,
            "elapsed": <seconds float>,
        }
    """
    task_name = task_entry.get("task", "unknown")
    t0 = time.monotonic()

    for attempt in range(1, max_retries + 1):
        try:
            log.info("[%s] attempt %d/%d", task_name, attempt, max_retries)
            raw = extraction(client, task_entry, contract_text)
            payload = parse_response(raw, task_name)

            elapsed = time.monotonic() - t0
            log.info("[%s] ✓ done in %.1fs", task_name, elapsed)

            status = "error" if isinstance(payload, dict) and "error" in payload else "ok"
            return {"task": task_name, "status": status, "payload": payload, "elapsed": elapsed}

        except Exception as exc:
            wait = backoff_base ** attempt
            log.warning("[%s] attempt %d failed (%s); retrying in %.0fs", task_name, attempt, exc, wait)
            if attempt < max_retries:
                time.sleep(wait)

    elapsed = time.monotonic() - t0
    log.error("[%s] all %d attempts failed", task_name, max_retries)
    return {
        "task": task_name,
        "status": "error",
        "payload": {"error": "max_retries_exceeded"},
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Parallel pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    client: LLMClient,
    llmcalls: List[Dict[str, Any]],
    contract_text: str,
    max_workers: Optional[int] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Run all tasks in parallel.

    max_workers defaults to min(len(llmcalls), 5) — enough concurrency to
    saturate typical API rate limits without hammering them.

    Returns a dict keyed by task name:
        {
            "Extract leaseJudgment Info": { "status": "ok", "payload": {...}, "elapsed": 4.2 },
            ...
        }
    """
    n = len(llmcalls)
    workers = max_workers or min(n, 5)
    log.info("Starting pipeline: %d tasks, %d workers", n, workers)

    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(run_task, client, task_entry, contract_text, max_retries): task_entry
            for task_entry in llmcalls
        }

        for future in as_completed(future_to_task):
            result = future.result()
            results[result["task"]] = result

    # Summary
    ok_count = sum(1 for r in results.values() if r["status"] == "ok")
    log.info("Pipeline complete: %d/%d tasks succeeded", ok_count, n)

    return results


# ---------------------------------------------------------------------------
# Convenience: flatten results to {task_name: payload} for downstream use
# ---------------------------------------------------------------------------

def flatten_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the status/elapsed envelope — returns just task → parsed payload."""
    return {name: r["payload"] for name, r in results.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="IFRS 16 extraction pipeline")
    parser.add_argument("--contract",  default="contract.txt",       help="Path to contract text file")
    parser.add_argument("--tasks",     default="llmcalls.json",      help="Path to llmcalls JSON")
    parser.add_argument("--output",    default="results.json",       help="Path for output JSON")
    parser.add_argument("--workers",   type=int, default=None,       help="Thread pool size (default: min(tasks, 5))")
    parser.add_argument("--retries",   type=int, default=3,          help="Max retries per task")
    parser.add_argument("--flat",      action="store_true",          help="Output flattened payload only (no status/elapsed)")
    args = parser.parse_args()

    # Load inputs
    contract_text = pathlib.Path(args.contract).read_text(encoding="utf-8")
    llmcalls      = json.loads(pathlib.Path(args.tasks).read_text(encoding="utf-8"))

    client = LLMClient(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],   
        deployment_name=os.environ["LLM_MODEL"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


    # Run
    results = run_pipeline(
        client=client,
        llmcalls=llmcalls,
        contract_text=contract_text,
        max_workers=args.workers,
        max_retries=args.retries,
    )

    # Write output
    output = flatten_results(results) if args.flat else results
    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Results written to %s", out_path)