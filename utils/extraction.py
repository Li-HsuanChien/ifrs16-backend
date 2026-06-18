"""
extraction.py — IFRS 16 contract extraction pipeline

Architecture
------------
Each topic runs as a two-step task: an extractioncalls task pulls the relevant
text splices out of the full contract, then its paired parsecalls task evaluates
the structured fields from just those splices.  Topics are independent of one
another, so the two-step tasks run in parallel via a ThreadPoolExecutor.  Each
step is wrapped with retry logic so a transient API failure doesn't abort the
whole run.

Flow:
  contract.txt ──────────────┐
                             ├──► [ThreadPoolExecutor]  (one task per topic)
  extractioncalls.json ──┐    │
  parsecalls.json ───────┴────┘         │
                  (paired by schema key)│
                            ├─ leaseJudgment : extract(splices) ─► parse(eval)
                            ├─ leaseTerms    : extract(splices) ─► parse(eval)
                            ├─ paymentTerms  : extract(splices) ─► parse(eval)
                            ├─ costSeparation: extract(splices) ─► parse(eval)
                            └─ discountRate  : extract(splices) ─► parse(eval)
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

from dotenv import load_dotenv

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
        if "choices" in raw:
            choice = raw["choices"][0]
            # Azure signals refusal or content filter here
            finish_reason = choice.get("finish_reason")
            if finish_reason not in ("stop", "length", None):
                return {"error": f"unexpected_finish_reason:{finish_reason}", "task": task_name}
            text = choice["message"]["content"]
        elif "content" in raw:
            text = raw["content"][0]["text"]
        else:
            return {"error": "unrecognised_envelope", "raw": raw, "task": task_name}

        if not text or not text.strip():
            return {"error": "empty_content", "task": task_name}

        payload = json.loads(text)

        # An empty object/array is a model failure, not a valid extraction
        if not payload:
            return {"error": "empty_payload", "task": task_name}

        return payload

    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        log.warning("[%s] parse_response failed: %s", task_name, exc)
        return {"error": str(exc), "raw_text": raw, "task": task_name}

def validate_payload(payload: Any, task_entry: Dict[str, Any], task_name: str) -> Optional[Dict]:
    """
    Lightweight structural check: verifies the top-level required keys from
    the outputSchema are present in the payload.
    Returns an error dict if invalid, None if ok.
    """
    schema = task_entry.get("outputSchema")
    if not schema:
        return None  # no schema to validate against

    required_keys = schema.get("required", [])
    if not required_keys:
        return None

    if not isinstance(payload, dict):
        return {"error": "payload_not_object", "task": task_name, "got": type(payload).__name__}

    missing = [k for k in required_keys if k not in payload]
    if missing:
        return {"error": "missing_required_keys", "task": task_name, "missing": missing}

    return None
# ---------------------------------------------------------------------------
# Task pairing (extractioncalls ↔ parsecalls)
# ---------------------------------------------------------------------------

def _schema_key(task_entry: Dict[str, Any]) -> str:
    """
    Stable identifier shared between an extraction task and its parse
    counterpart: the first top-level required key of the outputSchema
    (e.g. "LeaseJudgment").  Falls back to the task name.
    """
    schema = task_entry.get("outputSchema") or {}
    required = schema.get("required") or []
    return required[0] if required else task_entry.get("task", "unknown")


def pair_tasks(
    extractcalls: List[Dict[str, Any]],
    parsecalls: List[Dict[str, Any]],
) -> List[tuple]:
    """
    Pair each parse task with its extraction task by shared schema key.

    Returns a list of (extract_entry, parse_entry) tuples in parsecalls order.
    Parse tasks with no matching extraction task are skipped with a warning.
    """
    extract_by_key = {_schema_key(e): e for e in extractcalls}
    pairs: List[tuple] = []
    for parse_entry in parsecalls:
        key = _schema_key(parse_entry)
        extract_entry = extract_by_key.get(key)
        if extract_entry is None:
            log.warning(
                "No extraction task for parse task '%s' (key=%s); skipping",
                parse_entry.get("task"), key,
            )
            continue
        pairs.append((extract_entry, parse_entry))
    return pairs


# ---------------------------------------------------------------------------
# Single LLM call with retry (one pipeline step)
# ---------------------------------------------------------------------------

def _invoke_with_retry(
    client: LLMClient,
    task_entry: Dict[str, Any],
    input_text: str,
    max_retries: int,
    backoff_base: float,
) -> Any:
    """
    Run one LLM call (a single pipeline step) with retry + schema validation.
    Returns the parsed payload, or raises after exhausting retries.
    """
    task_name = task_entry.get("task", "unknown")

    for attempt in range(1, max_retries + 1):
        try:
            log.info("[%s] attempt %d/%d", task_name, attempt, max_retries)
            raw = extraction(client, task_entry, input_text)
            payload = parse_response(raw, task_name)
            schema_error = validate_payload(payload, task_entry, task_name)
            if schema_error:
                raise ValueError(schema_error["error"])
            if isinstance(payload, dict) and "error" in payload:
                raise ValueError(payload["error"])
            return payload

        except Exception as exc:
            wait = backoff_base ** attempt
            log.warning("[%s] attempt %d failed (%s); retrying in %.0fs", task_name, attempt, exc, wait)
            if attempt < max_retries:
                time.sleep(wait)

    raise RuntimeError("max_retries_exceeded")


# ---------------------------------------------------------------------------
# Two-step task runner (extract → parse)
# ---------------------------------------------------------------------------

def run_task(
    client: LLMClient,
    extract_entry: Dict[str, Any],
    parse_entry: Dict[str, Any],
    contract_text: str,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> Dict[str, Any]:
    """
    Run the two-step pipeline for one topic:

      Step 1 (extract): pull the relevant text splices out of the full contract
                        using `extract_entry` (extractioncalls).
      Step 2 (parse):   evaluate the structured fields from those splices using
                        `parse_entry` (parsecalls).

    Each step retries independently with exponential backoff.  Returns:

        {
            "task":      "Parse leaseJudgment Info",
            "status":    "ok" | "error",
            "payload":   <final parsed JSON> | <error dict>,
            "extracted": <step-1 content splices>,   (present on success)
            "elapsed":   <seconds float>,
        }
    """
    task_name = parse_entry.get("task", "unknown")
    t0 = time.monotonic()

    try:
        # Step 1 — extract relevant splices from the full contract text
        extracted = _invoke_with_retry(
            client, extract_entry, contract_text, max_retries, backoff_base
        )

        # Step 2 — parse/evaluate from the extracted splices
        parse_input = json.dumps(extracted, ensure_ascii=False)
        with open("verifi.txt", "a", encoding="utf-8") as file:
            file.write(",\n")
            file.write(parse_input)
        payload = _invoke_with_retry(
            client, parse_entry, parse_input, max_retries, backoff_base
        )

        elapsed = time.monotonic() - t0
        log.info("[%s] ✓ done in %.1fs", task_name, elapsed)
        return {
            "task": task_name,
            "status": "ok",
            "payload": payload,
            "elapsed": elapsed,
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.error("[%s] failed after retries: %s", task_name, exc)
        return {
            "task": task_name,
            "status": "error",
            "payload": {"error": str(exc)},
            "elapsed": elapsed,
        }


# ---------------------------------------------------------------------------
# Parallel pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    client: LLMClient,
    extractcalls: List[Dict[str, Any]],
    parsecalls: List[Dict[str, Any]],
    contract_text: str,
    max_workers: Optional[int] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Run the two-step extract→parse pipeline for every paired topic in parallel.

    `extractcalls` and `parsecalls` are paired by shared schema key (see
    pair_tasks); each pair runs as one two-step task.  max_workers defaults to
    min(pairs, 5) — enough concurrency to saturate typical API rate limits
    without hammering them.

    Returns a dict keyed by parse-task name:
        {
            "Parse leaseJudgment Info": { "status": "ok", "payload": {...}, "elapsed": 4.2 },
            ...
        }
    """
    pairs = pair_tasks(extractcalls, parsecalls)
    n = len(pairs)
    workers = max_workers or (min(n, 5) if n else 1)
    log.info("Starting pipeline: %d paired tasks, %d workers", n, workers)

    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(run_task, client, extract_entry, parse_entry,
                        contract_text, max_retries): parse_entry
            for (extract_entry, parse_entry) in pairs
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
    parser = argparse.ArgumentParser(description="IFRS 16 two-step extraction pipeline")
    parser.add_argument("--contract",      default="contract.txt",        help="Path to contract text file")
    parser.add_argument("--extract-tasks", default="extractioncalls.json", help="Path to extractioncalls JSON (step 1)")
    parser.add_argument("--parse-tasks",   default="parsecalls.json",      help="Path to parsecalls JSON (step 2)")
    parser.add_argument("--output",        default="results.json",        help="Path for output JSON")
    parser.add_argument("--workers",       type=int, default=None,        help="Thread pool size (default: min(tasks, 5))")
    parser.add_argument("--retries",       type=int, default=3,           help="Max retries per step")
    parser.add_argument("--flat",          action="store_true",           help="Output flattened payload only (no status/elapsed)")
    args = parser.parse_args()

    # Load inputs
    contract_text = pathlib.Path(args.contract).read_text(encoding="utf-8")
    extractcalls  = json.loads(pathlib.Path(args.extract_tasks).read_text(encoding="utf-8"))
    parsecalls    = json.loads(pathlib.Path(args.parse_tasks).read_text(encoding="utf-8"))

    client = LLMClient(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment_name=os.environ["LLM_MODEL"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


    # Run
    results = run_pipeline(
        client=client,
        extractcalls=extractcalls,
        parsecalls=parsecalls,
        contract_text=contract_text,
        max_workers=args.workers,
        max_retries=args.retries,
    )

    # Write output
    output = flatten_results(results) if args.flat else results
    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Results written to %s", out_path)