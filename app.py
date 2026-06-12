"""
api.py — IFRS 16 PDF extraction API

Endpoints
---------
POST /extract          — blocking; returns full JSON when pipeline finishes
POST /extract/stream   — SSE; pushes a status event for every pipeline stage
                         and each LLM task as it completes

SSE event shape (newline-delimited JSON, one object per line):
    {"event": "cv_started"}
    {"event": "cv_done",       "chars": 14200, "pages_failed": 0}
    {"event": "llm_started",   "tasks": 5}
    {"event": "task_done",     "task": "leaseJudgment", "status": "ok",  "elapsed": 3.1}
    {"event": "task_done",     "task": "paymentTerms",  "status": "ok",  "elapsed": 4.8}
    {"event": "task_error",    "task": "discountRate",  "error": "..."}
    {"event": "done",          "result": { ...full payload... }}
    {"event": "error",         "detail": "..."}        ← only on fatal failure

Query parameters (all optional, same for both endpoints):
    mode        ocr | document_ai   (default: ocr)
    backend     easyocr | azure     (default: easyocr; only used with document_ai)
    dpi         int                 (default: 300)
    preprocess  bool                (default: false)
    workers     int                 (default: min(tasks, 5))
    retries     int                 (default: 3)
    flat        bool                (default: false)

Environment variables required (.env):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_KEY
    AZURE_OPENAI_API_VERSION
    LLM_MODEL
    LLMCALLS_PATH   (optional, default: llmcalls.json)

    # Only when mode=document_ai and backend=azure:
    AZURE_DOCUMENTAI_ENDPOINT
    AZURE_DOCUMENTAI_KEY
"""

import asyncio
import json
import logging
import os
import pathlib
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from utils.docsextraction import ExtractionMode
from utils.docsextraction import run as docs_run
from utils.extraction import flatten_results, run_pipeline, run_task
from utils.llmapi import LLMClient

load_dotenv()

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
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="IFRS 16 Extraction API",
    description="Upload a lease PDF → get structured IFRS 16 JSON back.",
    version="1.0.0",
)
origins = ["http://localhost:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Shared state — built once at startup
# ---------------------------------------------------------------------------
_llm_client: Optional[LLMClient] = None
_llmcalls:   Optional[list]      = None


@app.on_event("startup")
def _startup() -> None:
    global _llm_client, _llmcalls

    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY",
                 "AZURE_OPENAI_API_VERSION", "LLM_MODEL"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")

    _llm_client = LLMClient(
        azure_endpoint  = os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment_name = os.environ["LLM_MODEL"],
        api_key         = os.environ["AZURE_OPENAI_KEY"],
        api_version     = os.environ["AZURE_OPENAI_API_VERSION"],
    )
    log.info("LLMClient initialised (deployment=%s)", os.environ["LLM_MODEL"])

    llmcalls_path = pathlib.Path(os.getenv("LLMCALLS_PATH", "llmcalls.json"))
    if not llmcalls_path.exists():
        raise RuntimeError(f"llmcalls.json not found at: {llmcalls_path.resolve()}")

    _llmcalls = json.loads(llmcalls_path.read_text(encoding="utf-8"))
    log.info("Loaded %d extraction task(s) from %s", len(_llmcalls), llmcalls_path)


# ---------------------------------------------------------------------------
# Shared query parameter defaults (used by both endpoints)
# ---------------------------------------------------------------------------
def _common_params(
    mode:       str           = Query("ocr",     enum=["ocr", "document_ai"]),
    backend:    str           = Query("easyocr", enum=["easyocr", "azure"]),
    dpi:        int           = Query(300,        ge=72,  le=600),
    preprocess: bool          = Query(False),
    workers:    Optional[int] = Query(None,       ge=1,   le=20),
    retries:    int           = Query(3,          ge=1,   le=10),
    flat:       bool          = Query(False),
):
    return dict(mode=mode, backend=backend, dpi=dpi,
                preprocess=preprocess, workers=workers,
                retries=retries, flat=flat)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sse(obj: dict) -> str:
    """Format one SSE data line."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _cv_stage(pdf_path: str, txt_path: str, mode, backend, dpi, preprocess) -> tuple[str, int]:
    """
    Run docsextraction synchronously (it is CPU-bound / I/O-bound, not async).
    Returns (contract_text, pages_failed_count).
    """
    success = docs_run(
        pdf_path    = pdf_path,
        output_path = txt_path,
        mode        = mode,
        backend     = backend,
        dpi         = dpi,
        preprocess  = preprocess,
        gpu         = False,
    )
    text = pathlib.Path(txt_path).read_text(encoding="utf-8").strip()
    # docs_run returns False when ≥1 page failed; we can't get the exact count
    # from its return value, so use 0/1 as a proxy.
    pages_failed = 0 if success else 1
    return text, pages_failed


# ---------------------------------------------------------------------------
# POST /extract  — blocking
# ---------------------------------------------------------------------------
@app.post("/extract", summary="Extract IFRS 16 data (blocking)")
async def extract(
    file:       UploadFile    = File(...),
    mode:       str           = Query("document_ai",     enum=["ocr", "document_ai"]),
    backend:    str           = Query("azure", enum=["easyocr", "azure"]),
    dpi:        int           = Query(300,        ge=72,  le=600),
    preprocess: bool          = Query(False),
    workers:    Optional[int] = Query(None,       ge=1,   le=20),
    retries:    int           = Query(3,          ge=1,   le=10),
    flat:       bool          = Query(False),
) -> JSONResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    extraction_mode = ExtractionMode.OCR if mode == "ocr" else ExtractionMode.DOCUMENT_AI

    with tempfile.TemporaryDirectory(prefix="ifrs16_api_") as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        pdf_bytes = await file.read()
        pathlib.Path(pdf_path).write_bytes(pdf_bytes)
        log.info("Received PDF: %s (%d bytes)", file.filename, len(pdf_bytes))

        txt_path = os.path.join(tmpdir, "extracted.txt")
        contract_text, _ = await asyncio.get_event_loop().run_in_executor(
            None, _cv_stage, pdf_path, txt_path, extraction_mode, backend, dpi, preprocess
        )

        if not contract_text:
            raise HTTPException(status_code=422, detail="CV extraction produced empty text.")

        results = run_pipeline(
            client        = _llm_client,
            llmcalls      = _llmcalls,
            contract_text = contract_text,
            max_workers   = workers,
            max_retries   = retries,
        )

    output = flatten_results(results) if flat else results
    
    # Add wrapper
    wrapped = {
        "contractInfo": {
            "fileName":     file.filename,
            "uploadDate":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "analysisDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        **output,
    }
    return JSONResponse(content=wrapped)


# ---------------------------------------------------------------------------
# POST /extract/stream  — SSE
# ---------------------------------------------------------------------------
@app.post("/extract/stream", summary="Extract IFRS 16 data (streaming SSE)")
async def extract_stream(
    file:       UploadFile    = File(...),
    mode:       str           = Query("document_ai",     enum=["ocr", "document_ai"]),
    backend:    str           = Query("azure", enum=["easyocr", "azure"]),
    dpi:        int           = Query(300,        ge=72,  le=600),
    preprocess: bool          = Query(False),
    workers:    Optional[int] = Query(None,       ge=1,   le=20),
    retries:    int           = Query(3,          ge=1,   le=10),
    flat:       bool          = Query(False),
) -> StreamingResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    extraction_mode = ExtractionMode.OCR if mode == "ocr" else ExtractionMode.DOCUMENT_AI

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        upload_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with tempfile.TemporaryDirectory(prefix="ifrs16_stream_") as tmpdir:

            # ----------------------------------------------------------------
            # Stage 1 — CV
            # ----------------------------------------------------------------
            yield _sse({"event": "cv_started", "filename": file.filename})

            pdf_path = os.path.join(tmpdir, "input.pdf")
            pathlib.Path(pdf_path).write_bytes(pdf_bytes)
            txt_path = os.path.join(tmpdir, "extracted.txt")

            try:
                contract_text, pages_failed = await loop.run_in_executor(
                    None,
                    _cv_stage,
                    pdf_path, txt_path,
                    extraction_mode, backend, dpi, preprocess,
                )
            except Exception as exc:
                log.exception("CV stage failed")
                yield _sse({"event": "error", "detail": f"CV stage failed: {exc}"})
                return

            if not contract_text:
                yield _sse({"event": "error", "detail": "CV extraction produced empty text."})
                return

            yield _sse({
                "event":        "cv_done",
                "chars":        len(contract_text),
                "pages_failed": pages_failed,
            })

            # ----------------------------------------------------------------
            # Stage 2 — LLM tasks (yield each as it finishes)
            # ----------------------------------------------------------------
            n       = len(_llmcalls)
            workers_count = workers or min(n, 5)
            yield _sse({"event": "llm_started", "tasks": n, "workers": workers_count})

            results: dict = {}

            # Run the thread pool in a background thread so we can yield
            # results incrementally without blocking the event loop.
            # We replicate the as_completed logic from run_pipeline here so
            # we can stream each result the moment it arrives.
            def _run_all():
                """Runs in a single executor thread; returns a queue of results."""
                import queue
                q: queue.Queue = queue.Queue()

                with ThreadPoolExecutor(max_workers=workers_count) as pool:
                    futures = {
                        pool.submit(run_task, _llm_client, task_entry,
                                    contract_text, retries): task_entry
                        for task_entry in _llmcalls
                    }
                    for future in as_completed(futures):
                        q.put(future.result())

                q.put(None)   # sentinel
                return q

            q = await loop.run_in_executor(None, _run_all)

            # Drain the queue as results arrive
            while True:
                try:
                    result = q.get_nowait()
                except Exception:
                    # Queue not ready yet — yield control and retry
                    await asyncio.sleep(0.05)
                    continue

                if result is None:
                    break   # sentinel — all tasks done

                task_name = result["task"]
                results[task_name] = result

                if result["status"] == "ok":
                    yield _sse({
                        "event":   "task_done",
                        "task":    task_name,
                        "status":  "ok",
                        "elapsed": round(result["elapsed"], 2),
                    })
                else:
                    yield _sse({
                        "event":   "task_error",
                        "task":    task_name,
                        "status":  "error",
                        "error":   result["payload"].get("error", "unknown"),
                        "elapsed": round(result["elapsed"], 2),
                    })

            # ----------------------------------------------------------------
            # Final event — full result payload
            # ----------------------------------------------------------------
            ok_count = sum(1 for r in results.values() if r["status"] == "ok")
            output   = flatten_results(results) if flat else results

            wrapped = {
                "contractInfo": {
                    "fileName":     file.filename,
                    "uploadDate":   upload_time, # ← when the PDF arrived
                    "analysisDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), # ← when analysis finished
                },
                **output,
            }

            yield _sse({
                "event":       "done",
                "tasks_ok":    ok_count,
                "tasks_total": n,
                "result":      wrapped,   # ← was just `output`
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "tasks_loaded": len(_llmcalls) if _llmcalls else 0}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)