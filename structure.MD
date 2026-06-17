# IFRS 16 Backend — Architecture & Structure

Backend service that turns an uploaded lease **PDF** into structured **IFRS 16**
JSON. It runs three stages: OCR/Document-AI text extraction → a two-step LLM
pipeline (extract relevant splices, then evaluate them against IFRS 16 rules) →
a wrapped JSON response.

---

## 1. Repository layout

```
ifrs16-backend/
├── app.py                  # FastAPI service — HTTP entry point & orchestration
├── extractioncalls.json    # Step-1 prompts: contract text → content splices
├── parsecalls.json         # Step-2 prompts: content splices → evaluated IFRS16 fields
├── utils/
│   ├── docsextraction.py   # Stage 1: PDF → text (OCR / Azure Document Intelligence)
│   ├── llmapi.py           # LLM client + prompt/message builders + single call
│   └── extraction.py       # Stage 2/3: two-step LLM pipeline orchestration
├── sampleRen0.pdf          # Sample input
├── .env                    # Secrets / config (not committed)
└── structure.MD            # This document
```

> `utils/` has no `__init__.py`; it works as a namespace package. `app.py`
> imports it as `utils.<module>`, and `utils/extraction.py` imports its sibling
> as `utils.llmapi`. Run the server from the **repo root** so these resolve.

---

## 2. End-to-end data flow

```
            ┌─────────────────────────────────────────────────────────────┐
  PDF  ───► │  app.py  POST /extract  (or /extract/stream)                 │
            └─────────────────────────────────────────────────────────────┘
                              │
        Stage 1               ▼  utils/docsextraction.py : run()
   ┌──────────────────────────────────────────────────────────────────┐
   │ PDF → PNG per page (poppler, parallel)                            │
   │ → optional preprocess (adaptive threshold)                       │
   │ → per page: EasyOCR or Azure Document Intelligence (parallel,    │
   │   retried up to 5× on empty result)                              │
   │ → merge pages into one contract_text string                     │
   └──────────────────────────────────────────────────────────────────┘
                              │  contract_text (str)
        Stage 2/3            ▼  utils/extraction.py : run_pipeline()
   ┌──────────────────────────────────────────────────────────────────┐
   │ pair extractioncalls.json ↔ parsecalls.json by schema key        │
   │ for each topic, in parallel (ThreadPoolExecutor):                │
   │   Step 1 (extract): full contract_text ─► content splices       │
   │                     using extractioncalls task                  │
   │   Step 2 (parse):   content splices ─► evaluated fields          │
   │                     using parsecalls task                       │
   │   each step: retry + schema validation                          │
   └──────────────────────────────────────────────────────────────────┘
                              │  results { task_name → envelope }
                              ▼
            ┌─────────────────────────────────────────────────────────────┐
            │  app.py wraps with contractInfo and returns JSON            │
            └─────────────────────────────────────────────────────────────┘
```

The **two-step** design (extract-then-parse) keeps each LLM call focused: the
extraction step only locates the relevant verbatim text per field; the parse step
only reasons over those small splices instead of the whole noisy OCR dump. This
improves accuracy and lets the two prompt sets evolve independently.

---

## 3. `app.py` — FastAPI service

### Endpoints

| Method | Path              | Description |
|--------|-------------------|-------------|
| POST   | `/extract`        | Blocking. Returns the full JSON once the pipeline finishes. |
| POST   | `/extract/stream` | Server-Sent Events. Emits a status event per stage and per LLM task as it completes. |
| GET    | `/health`         | Liveness + number of parse tasks loaded. |

### Query parameters (both POST endpoints)

| Param        | Default       | Meaning |
|--------------|---------------|---------|
| `mode`       | `document_ai` | `ocr` (flat EasyOCR) or `document_ai` (structure-aware). |
| `backend`    | `azure`       | `easyocr` or `azure` (only used when `mode=document_ai`). |
| `dpi`        | `300`         | Rasterisation resolution (72–600). |
| `preprocess` | `false`       | Adaptive-threshold the page image before extraction. |
| `workers`    | `None`        | LLM-task concurrency (defaults to `min(pairs, 5)`). |
| `retries`    | `3`           | Max retries **per LLM step**. |
| `flat`       | `false`       | Return only `task → payload` (drop status/elapsed envelope). |

### Startup (`_startup`)

- Validates required env vars (see §7) and builds a shared `LLMClient`.
- Loads **two** task files into module globals:
  - `_extractcalls` ← `EXTRACTIONCALLS_PATH` (default `extractioncalls.json`)
  - `_parsecalls`   ← `LLMCALLS_PATH` (default `parsecalls.json`)

### CV stage helper (`_cv_stage`)

Runs `docsextraction.run()` in a thread-pool executor (it is CPU/IO-bound, not
async) and returns `(contract_text, pages_failed)`. It does **not** pass
`max_workers`, so docsextraction auto-picks concurrency per backend.

### SSE event shape (`/extract/stream`)

```
{"event": "cv_started",  "filename": "..."}
{"event": "cv_done",     "chars": 14200, "pages_failed": 0}
{"event": "llm_started", "tasks": 5, "workers": 5}
{"event": "task_done",   "task": "Parse leaseJudgment Info", "status": "ok", "elapsed": 3.1}
{"event": "task_error",  "task": "...", "status": "error", "error": "...", "elapsed": 2.0}
{"event": "done",        "tasks_ok": 5, "tasks_total": 5, "result": { ...wrapped... }}
{"event": "error",       "detail": "..."}    # fatal failure only
```

The streaming endpoint pairs tasks (`pair_tasks`) and submits one **two-step**
`run_task` per pair to its own `ThreadPoolExecutor`, draining results onto a
queue so each task can be streamed the moment it finishes.

### Final response wrapper

Both endpoints wrap the pipeline output with a `contractInfo` header:

```json
{
  "contractInfo": { "fileName": "...", "uploadDate": "...", "analysisDate": "..." },
  "Parse leaseJudgment Info": { "status": "ok", "payload": { "LeaseJudgment": {...} }, "elapsed": 4.2 },
  "...": ...
}
```

With `flat=true`, each topic value is just its `payload`.

---

## 4. The two prompt files

Both files are JSON **arrays** of task objects sharing the same outer shape:

```jsonc
{
  "task":           "<human label>",
  "description":    "<one-line summary>",
  "system_prompts": "<the system instruction text>",
  "rules":          [ "B13:...", ... ]  | "",   // IFRS reference rules (leaseJudgment only)
  "examples":       [ { ...few-shot... } ],
  "outputSchema":   { ...JSON Schema (strict) ... }
}
```

### Pairing key

Extraction and parse tasks are matched by `outputSchema.required[0]` — the
PascalCase top-level key (`LeaseJudgment`, `LeaseTerms`, `PaymentTerms`,
`CostSeparation`, `DiscountRate`). Task **names differ on purpose**
(`"Extract leaseJudgment"` vs `"Parse leaseJudgment Info"`); pairing never relies
on the name. See `pair_tasks` / `_schema_key` in `extraction.py`.

### `extractioncalls.json` — Step 1 (locate, don't judge)

- System prompt instructs the model to **only pull verbatim source text** into a
  per-field `content` string — no classification, scoring, or date conversion.
- `outputSchema` mirrors the parse schema's structure but every leaf is just
  `{ "content": string }`.
- Example output (leaseJudgment):

```json
{
  "LeaseJudgment": {
    "isSpecificAsset":  { "content": "高雄市新興區復興一路8號4樓部份位置面積約342坪(下稱外櫃)" },
    "hasControlRight":  { "content": "未提及...享有該資產之效益。" },
    "assetReplaceable": { "content": "未提及決定資產的使用方式及用途" }
  }
}
```

### `parsecalls.json` — Step 2 (evaluate against IFRS 16)

- System prompt states the **input is the content-splice object** from step 1,
  then evaluates each field (`value`/`confidence`/`userConfirmed`, dates in ISO
  8601 with ROC→Gregorian conversion, lease/non-lease allocation, etc.).
- `rules` (leaseJudgment) carries the IFRS 16 reference clauses B13–B30, appended
  to the system prompt as a numbered "Reference Rules" block.
- Each `examples[*]` entry now carries a **`source`** field holding the
  content-splice object that step 1 would emit — this is the demonstrated *input*
  for the few-shot (see §5, `build_few_shot_messages`).

### Topics & output keys

| Topic          | Output key       | Shape  | Notable fields |
|----------------|------------------|--------|----------------|
| leaseJudgment  | `LeaseJudgment`  | object | isSpecificAsset, hasControlRight, assetReplaceable |
| leaseTerms     | `LeaseTerms`     | object | period start/end, nonCancellablePeriod, renewalOption, earlyTermination, prepayment, finalPayment |
| paymentTerms   | `PaymentTerms`   | array  | per-period: periodicRent, frequency, timing, dates, variableRent |
| costSeparation | `CostSeparation` | array  | per-period: leaseComponent / nonLeaseComponent split (% must total 100) |
| discountRate   | `DiscountRate`   | object | implicitRate, incrementalBorrowingRate |

---

## 5. `utils/llmapi.py` — LLM client & prompt assembly

### `LLMClient`

Thin wrapper over the **Azure OpenAI Chat Completions** REST endpoint
(`/openai/deployments/{deployment}/chat/completions?api-version=...`). Key points:

- System prompt is sent as the first `role: "system"` message (no top-level
  `system` field; deployment is in the URL, not the body).
- When an `outputSchema` is supplied it uses **structured outputs**
  (`response_format: { type: "json_schema", strict: true }`).
- `generate()` posts the assembled messages; raises on non-200.

### Prompt/message builders

| Function | Role |
|----------|------|
| `build_system_prompt(task)` | `system_prompts` + (optional) numbered `rules` block. |
| `build_few_shot_messages(task)` | One user/assistant pair per example. The example's **`source`** becomes the *user* turn (raw contract text for step 1, or the content object for step 2); the example **minus `source`** becomes the *assistant* turn (expected output). Falls back to a placeholder if no `source`. |
| `build_user_prompt(input_text, n)` | Wraps the live input in the same "extract from the following input" format. |
| `extraction(client, task, input_text)` | Assembles `system + few-shot + live input`, calls `generate()` with the task's `outputSchema`. Used for **both** steps — `input_text` is the contract for step 1, or the serialized splices for step 2. Returns the raw API envelope (or `None` on error). |

---

## 6. `utils/extraction.py` — pipeline orchestration

### Response handling

- `parse_response(raw, task)` — unwraps either the Azure/OpenAI
  (`choices[0].message.content`) or Anthropic (`content[0].text`) envelope,
  JSON-parses the inner text, and returns the payload or an error dict
  (handles refusals, empty content, empty payload, decode errors).
- `validate_payload(payload, task, name)` — lightweight check that the
  `outputSchema.required` top-level keys are present.

### Pairing & retry

- `_schema_key(task)` / `pair_tasks(extractcalls, parsecalls)` — match each parse
  task to its extraction task by schema key; unmatched parse tasks are skipped
  with a warning.
- `_invoke_with_retry(client, task, input_text, max_retries, backoff_base)` —
  runs **one** LLM call (one step) with exponential backoff; validates schema and
  treats an `error` payload as a retryable failure; raises after exhausting tries.

### Two-step task & fan-out

- `run_task(client, extract_entry, parse_entry, contract_text, ...)` —
  Step 1 extracts splices from the full contract; Step 2 feeds
  `json.dumps(splices)` into the parse task. Returns:

  ```jsonc
  {
    "task": "Parse leaseJudgment Info",
    "status": "ok" | "error",
    "payload": { ...evaluated... } | { "error": "..." },
    "extracted": { ...step-1 splices... },   // present on success
    "elapsed": 4.2
  }
  ```

- `run_pipeline(client, extractcalls, parsecalls, contract_text, max_workers, max_retries)`
  — pairs the lists and runs one two-step `run_task` per topic in parallel
  (`ThreadPoolExecutor`, default `min(pairs, 5)` workers). Returns
  `{ parse_task_name → envelope }`.
- `flatten_results(results)` — strip envelopes down to `{ task_name → payload }`.

CLI: `python utils/extraction.py --contract contract.txt --extract-tasks extractioncalls.json --parse-tasks parsecalls.json --output results.json [--workers N] [--retries N] [--flat]`.

---

## 7. `utils/docsextraction.py` — PDF → text (Stage 1)

### Stages

1. **Rasterise** (`rasterise`) — `pdf2image.convert_from_path` with poppler
   `thread_count` (parallel page rendering) → per-page PNGs.
2. **Preprocess** (optional, `preprocess_image`) — OpenCV adaptive threshold for
   noisy scans.
3. **Extract per page**:
   - `ocr_image` (EasyOCR, flat text), or
   - `document_ai_image` → `_document_ai_easyocr` / `_document_ai_azure`
     (Azure Document Intelligence `prebuilt-read`, returns per-line blocks).
4. **Merge** — `merge_ocr` (flat concat) or `merge_document_ai` (per-page labelled
   block sections). Failed pages get a `[PAGE n EXTRACTION FAILED]` placeholder.

### Performance & reliability

- **Parallel pages** — extraction fans out across a `ThreadPoolExecutor`, writing
  results back by index so **page order is preserved**.
  - **Azure** backend → parallel (default up to 8 workers; network/IO-bound).
  - **EasyOCR** → forced **sequential** (shared PyTorch model isn't thread-safe;
    a single GPU serialises anyway). Explicit `workers>1` is warned and downgraded.
- **Per-page retry** — `_is_valid_extraction` flags a page as invalid when it
  produced no text (empty string for OCR, no blocks for Document AI, or `None`).
  `_extract_page` retries up to `max_page_retries` (default **5**) with
  exponential backoff before giving up. The final success/failure count uses this
  validity check.

`run(...)` signature highlights: `mode`, `backend`, `dpi`, `preprocess`,
`max_workers`, `max_page_retries`. CLI mirrors these (`--workers`, `--page-retries`, …).

### Required env (Azure backend)

```
AZURE_DOCUMENTAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENTAI_KEY=<key>
```

---

## 8. Environment variables (`.env`)

| Variable | Used by | Purpose |
|----------|---------|---------|
| `AZURE_OPENAI_ENDPOINT` | llmapi | Azure OpenAI resource endpoint. |
| `AZURE_OPENAI_KEY` | llmapi | Azure OpenAI API key. |
| `AZURE_OPENAI_API_VERSION` | llmapi | API version string. |
| `LLM_MODEL` | llmapi | Deployment name used in the URL path. |
| `EXTRACTIONCALLS_PATH` | app | Step-1 task file (default `extractioncalls.json`). |
| `LLMCALLS_PATH` | app | Step-2 task file (default `parsecalls.json`). |
| `AZURE_DOCUMENTAI_ENDPOINT` | docsextraction | Document Intelligence endpoint (azure backend). |
| `AZURE_DOCUMENTAI_KEY` | docsextraction | Document Intelligence key (azure backend). |

---

## 9. Running

```bash
# from repo root, with venv active and .env populated
uvicorn app:app --host 0.0.0.0 --port 8000 --reload    # or: python app.py

# CLI-only pipeline (skips the API):
python utils/docsextraction.py --input sampleRen0.pdf --output contract.txt --mode document_ai --backend azure
python utils/extraction.py     --contract contract.txt --output results.json
```

CORS currently allows `http://localhost:8080` (the frontend dev origin).

---

## 10. Target output shape (reference)

The per-topic `payload` values combine into the IFRS 16 object the frontend
consumes. Representative shape (values illustrative):

```jsonc
{
  "LeaseJudgment": {
    "isSpecificAsset":  { "value": true,  "content": "...", "confidence": 95, "userConfirmed": false },
    "hasControlRight":  { "value": true,  "content": "...", "confidence": 90, "userConfirmed": false },
    "assetReplaceable": { "value": true,  "content": "...", "confidence": 85, "userConfirmed": false }
  },
  "LeaseTerms": {
    "periodStartDate": { "value": "2024-01-01T00:00:00.000Z", "content": "...", "confidence": 95, "userConfirmed": false },
    "periodEndDate":   { "value": "2026-12-31T00:00:00.000Z", "content": "...", "confidence": 95, "userConfirmed": false },
    "nonCancellablePeriod": { "value": 36, "content": "...", "confidence": 90, "userConfirmed": false },
    "renewalOption":   { "hasOption": true, "period": 24, "content": "...", "confidence": 80, "userConfirmed": false },
    "reasonablyExpectedToExercise": { "value": true, "reason": "...", "confidence": 75, "userConfirmed": false },
    "earlyTermination":{ "hasClause": false, "penalty": null, "penaltyType": null, "content": "...", "confidence": 85, "userConfirmed": false },
    "prepayment":      { "hasPrepayment": false, "amount": null, "content": "...", "confidence": 80, "userConfirmed": false },
    "finalPayment":    { "hasPayment": false, "amount": null, "content": "...", "confidence": 80, "userConfirmed": false }
  },
  "PaymentTerms": [
    {
      "periodicRent": { "amount": 50000, "content": "...", "confidence": 95, "userConfirmed": false },
      "frequency":    { "value": "monthly", "content": "...", "confidence": 90, "userConfirmed": false },
      "timing":       { "value": "advance", "content": "...", "confidence": 90, "userConfirmed": false },
      "periodStartDate": { "value": "2024-01-01T00:00:00.000Z", "content": "...", "confidence": 95, "userConfirmed": false },
      "periodEndDate":   { "value": "2026-12-31T00:00:00.000Z", "content": "...", "confidence": 95, "userConfirmed": false },
      "variableRent": { "hasVariable": false, "type": null, "content": "...", "confidence": 85, "userConfirmed": false },
      "variableRentPredictable": { "isPredictable": null, "reason": "...", "confidence": 0, "userConfirmed": false }
    }
  ],
  "CostSeparation": [
    {
      "periodStartDate": { "value": "2024-01-01T00:00:00.000Z", "content": "...", "confidence": 95, "userConfirmed": false },
      "periodEndDate":   { "value": "2026-12-31T00:00:00.000Z", "content": "...", "confidence": 95, "userConfirmed": false },
      "leaseComponent":    { "amount": 47000, "percentage": 94, "allocationMethod": "relative_standalone_price", "contractReference": "第5條", "confidence": 85, "userConfirmed": false },
      "nonLeaseComponent": { "amount": 3000,  "percentage": 6,  "allocationMethod": "relative_standalone_price", "contractReference": "第7條", "confidence": 80, "userConfirmed": false }
    }
  ],
  "DiscountRate": {
    "incrementalBorrowingRate": { "rate": 0.045, "source": "...", "currency": "TWD", "confidence": 80, "userConfirmed": false },
    "implicitRate":             { "rate": null,  "source": "...", "currency": "TWD", "confidence": 0,  "userConfirmed": false }
  }
}
```

> `LeaseJudgment` / `DiscountRate` are objects; `PaymentTerms` / `CostSeparation`
> are arrays (one entry per rate period). Shapes above are taken directly from the
> `outputSchema` and `examples` in `parsecalls.json`.
