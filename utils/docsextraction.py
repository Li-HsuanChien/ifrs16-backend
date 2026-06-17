"""
docsextraction.py

Pipeline for extracting text from PDF files via OCR or Document AI.

Stages:
    1. PDF  → per-page PNG images  (pdf2image)
    2. PNG  → text                 (EasyOCR  |  Azure Document Intelligence)
    3. Text → single output file   (flat concat for OCR, structured for Document AI)

Usage (CLI):
    python docsextraction.py input.pdf output.txt --mode ocr --preprocess --dpi 300
    python docsextraction.py input.pdf output.txt --mode document_ai --backend azure

Usage (library):
    from docsextraction import run, ExtractionMode

    run("input.pdf", "output.txt", mode=ExtractionMode.DOCUMENT_AI, backend="azure")

Azure setup (.env):
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
    AZURE_DOCUMENT_INTELLIGENCE_KEY=<your-key>
"""

import argparse
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from typing import Callable, Optional

import cv2
import easyocr
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class ExtractionMode(Enum):
    OCR         = auto()   # Flat text; pages concatenated.
    DOCUMENT_AI = auto()   # Structure-aware; per-page blocks kept separate.

OcrResult  = list[tuple]                  # [(bbox, text, confidence), ...]
OcrApiFunc = Callable[[str], OcrResult]


# ---------------------------------------------------------------------------
# Stage 1 – PDF → PNG paths
# ---------------------------------------------------------------------------
def rasterise(pdf_path: str, tmpdir: str, dpi: int = 300, thread_count: int = 4) -> list[str]:
    """
    Convert each PDF page to a PNG inside tmpdir; return sorted PNG paths.

    `thread_count` is passed through to poppler (pdf2image) so page rendering
    runs in parallel across cores.
    """
    log.info("Rasterising %s at %d DPI (thread_count=%d) …", pdf_path, dpi, thread_count)
    pages = convert_from_path(pdf_path, dpi=dpi, thread_count=thread_count)
    png_paths: list[str] = []
    for i, page in enumerate(pages, start=1):
        png_path = os.path.join(tmpdir, f"page_{i:04d}.png")
        page.save(png_path, "PNG")
        png_paths.append(png_path)
    log.info("Rasterised %d page(s).", len(png_paths))
    return png_paths


# ---------------------------------------------------------------------------
# Stage 1b – optional image preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(input_path: str, output_path: str) -> bool:
    """Apply adaptive thresholding to improve OCR on noisy scans."""
    try:
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"cv2 could not open: {input_path}")
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2,
        )
        cv2.imwrite(output_path, binary)
        return True
    except Exception:
        log.exception("Preprocessing failed for %s", input_path)
        return False


# ---------------------------------------------------------------------------
# Azure Document Intelligence client factory
# ---------------------------------------------------------------------------
def _make_azure_client():
    """
    Construct an Azure DocumentAnalysisClient from environment variables.

    Required .env keys:
        AZURE_DOCUMENTAI_ENDPOINT
        AZURE_DOCUMENTAI_KEY

    Raises RuntimeError if either variable is missing.
    Raises ImportError if the Azure SDK is not installed.
    """
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError as e:
        raise ImportError(
            "Azure SDK not installed. Run: "
            "pip install azure-ai-documentintelligence azure-core"
        ) from e

    endpoint = os.getenv("AZURE_DOCUMENTAI_ENDPOINT")
    key      = os.getenv("AZURE_DOCUMENTAI_KEY")

    if not endpoint or not key:
        raise RuntimeError(
            "Azure credentials missing. Set AZURE_DOCUMENTAI_ENDPOINT "
            "and AZURE_DOCUMENTAI_KEY in your .env file."
        )

    return DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))


# ---------------------------------------------------------------------------
# Stage 2a – OCR extraction  (EasyOCR)
# ---------------------------------------------------------------------------
def ocr_image(image_path: str, ocr_api: OcrApiFunc) -> Optional[str]:
    """Return flat extracted text from one image, or None on failure."""
    try:
        lines: list[str] = []
        for _bbox, text, _conf in ocr_api(image_path):
            cleaned = re.sub(r'[a-zA-Z.!\'"^@#%/\-{}\[\]]', "", text)
            if cleaned.strip():
                lines.append(cleaned)
        result = "".join(lines)
        log.info("OCR: %s → %d chars", image_path, len(result))
        return result
    except Exception:
        log.exception("OCR failed for %s", image_path)
        return None


# ---------------------------------------------------------------------------
# Stage 2b – Document AI extraction
# ---------------------------------------------------------------------------
def _document_ai_easyocr(image_path: str, ocr_api: OcrApiFunc) -> Optional[dict]:
    """EasyOCR-backed structured extraction."""
    try:
        blocks = []
        for bbox, text, confidence in ocr_api(image_path):
            cleaned = re.sub(r'[a-zA-Z.!\'"^@#%/\-{}\[\]]', "", text)
            if cleaned.strip():
                blocks.append({"text": cleaned, "bbox": bbox, "confidence": confidence})
        log.info("Document AI (easyocr): %s → %d blocks", image_path, len(blocks))
        return {"blocks": blocks}
    except Exception:
        log.exception("Document AI (easyocr) failed for %s", image_path)
        return None


def _document_ai_azure(image_path: str, azure_client) -> Optional[dict]:
    """
    Azure Document Intelligence backed structured extraction.

    Uses the prebuilt-read model. Swap the model_id string for
    'prebuilt-layout', 'prebuilt-invoice', etc. as needed.
    """
    try:
        with open(image_path, "rb") as f:
            poller = azure_client.begin_analyze_document(
                model_id="prebuilt-read",
                body=f,
                content_type="image/png",
            )
        result = poller.result()

        blocks = []
        for page in result.pages:
            for line in (page.lines or []):
                text = line.content.strip()
                if not text:
                    continue
                # Polygon is a flat list [x0,y0,x1,y1,...]; package as bbox.
                bbox = line.polygon or []
                blocks.append({
                    "text":       text,
                    "bbox":       bbox,
                    "confidence": getattr(line, "confidence", None),
                })

        log.info("Document AI (azure): %s → %d blocks", image_path, len(blocks))
        return {"blocks": blocks}
    except Exception:
        log.exception("Document AI (azure) failed for %s", image_path)
        return None


def document_ai_image(
    image_path:   str,
    backend:      str,
    ocr_api:      Optional[OcrApiFunc] = None,
    azure_client = None,
) -> Optional[dict]:
    """
    Dispatch to the appropriate Document AI backend.

    Args:
        image_path:   Path to the PNG to analyse.
        backend:      "easyocr" or "azure".
        ocr_api:      Required when backend="easyocr".
        azure_client: Required when backend="azure".
    """
    if backend == "azure":
        if azure_client is None:
            raise ValueError("azure_client must be provided when backend='azure'")
        return _document_ai_azure(image_path, azure_client)

    if backend == "easyocr":
        if ocr_api is None:
            raise ValueError("ocr_api must be provided when backend='easyocr'")
        return _document_ai_easyocr(image_path, ocr_api)

    raise ValueError(f"Unknown Document AI backend: {backend!r}. Choose 'easyocr' or 'azure'.")


# ---------------------------------------------------------------------------
# Stage 2c – Result validity
# ---------------------------------------------------------------------------
def _is_valid_extraction(result, mode: ExtractionMode) -> bool:
    """
    A page result is valid only if extraction actually produced text.

    - OCR mode:         a non-empty, non-whitespace string.
    - Document AI mode: a dict with at least one block.

    None (an extraction failure) is always invalid.
    """
    if result is None:
        return False
    if mode is ExtractionMode.OCR:
        return bool(str(result).strip())
    # Document AI: must have at least one extracted block
    return isinstance(result, dict) and bool(result.get("blocks"))


# ---------------------------------------------------------------------------
# Stage 3 – Merge
# ---------------------------------------------------------------------------
def merge_ocr(page_texts: list[Optional[str]]) -> str:
    """Concatenate all pages into one flat string."""
    parts: list[str] = []
    for i, text in enumerate(page_texts, start=1):
        if text is None:
            log.warning("Page %d failed; inserting placeholder.", i)
            parts.append(f"[PAGE {i} EXTRACTION FAILED]\n")
        else:
            parts.append(text)
    return "".join(parts)


def merge_document_ai(page_results: list[Optional[dict]]) -> str:
    """Render each page as a labelled block section."""
    sections: list[str] = []
    for i, result in enumerate(page_results, start=1):
        header = f"{'='*60}\nPAGE {i}\n{'='*60}\n"
        if result is None:
            sections.append(header + f"[PAGE {i} EXTRACTION FAILED]\n")
            continue
        block_lines = "\n".join(
            f"  [{j+1}] {block['text']}"
            for j, block in enumerate(result["blocks"])
        )
        sections.append(header + block_lines + "\n")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------
def run(
    pdf_path:    str,
    output_path: str,
    mode:        ExtractionMode = ExtractionMode.DOCUMENT_AI,
    backend:     str            = "azure",
    languages:   list[str]      = None,
    gpu:         bool           = True,
    dpi:         int            = 300,
    preprocess:  bool           = False,
    max_workers: Optional[int]  = None,
    max_page_retries: int       = 5,
) -> bool:
    """
    Full pipeline: PDF → PNGs → text → output file.

    Args:
        pdf_path:    Source PDF.
        output_path: Destination text file.
        mode:        ExtractionMode.OCR or ExtractionMode.DOCUMENT_AI.
        backend:     "easyocr" (default) or "azure".
                     Only consulted when mode=DOCUMENT_AI.
        languages:   EasyOCR language codes (ignored when backend="azure").
        gpu:         Use GPU for EasyOCR (ignored when backend="azure").
        dpi:         Rasterisation resolution.
        preprocess:  Apply adaptive threshold before extraction.
        max_workers: Page-extraction concurrency.  When None, defaults to
                     parallel for the Azure backend (network-bound) and
                     sequential for EasyOCR (the torch model is not reliably
                     thread-safe and a single GPU serialises anyway).
        max_page_retries: Attempts per page before giving up.  A page whose
                     extraction returns no text/blocks is retried (with
                     exponential backoff) instead of silently failing.

    Returns True if all pages extracted cleanly, False if any page failed.
    """
    # -- Initialise backend clients only as needed ---------------------------
    reader       = None
    azure_client = None

    needs_easyocr = mode is ExtractionMode.OCR or (
        mode is ExtractionMode.DOCUMENT_AI and backend == "easyocr"
    )
    needs_azure = mode is ExtractionMode.DOCUMENT_AI and backend == "azure"

    if needs_easyocr:
        log.info("Loading EasyOCR (languages=%s, gpu=%s) …", languages or ["ch_tra", "en"], gpu)
        reader = easyocr.Reader(languages or ["ch_tra", "en"], gpu=gpu)

    if needs_azure:
        log.info("Initialising Azure Document Intelligence client …")
        azure_client = _make_azure_client()   # raises clearly if .env is missing

    # -- Rasterise and extract -----------------------------------------------
    cpu = os.cpu_count() or 1
    with tempfile.TemporaryDirectory(prefix="pdf_extract_") as tmpdir:
        png_paths = rasterise(pdf_path, tmpdir, dpi=dpi, thread_count=min(4, cpu))
        if not png_paths:
            log.error("No pages found in %s", pdf_path)
            return False

        n = len(png_paths)

        # Decide page-extraction concurrency.
        #   Azure  → network/IO-bound, safe to fan out.
        #   EasyOCR→ shared torch model is not thread-safe; keep sequential.
        if max_workers is not None:
            workers = max(1, min(max_workers, n))
            if needs_easyocr and workers > 1:
                log.warning(
                    "EasyOCR is not thread-safe; forcing sequential extraction "
                    "(requested max_workers=%d).", max_workers
                )
                workers = 1
        else:
            workers = min(8, n) if needs_azure else 1

        def _extract_page(i: int, png_path: str) -> Optional[object]:
            """
            Preprocess (optional) then extract one page, retrying on an invalid
            result up to `max_page_retries` times with exponential backoff.
            Returns the last result (valid, or the final invalid one).
            """
            target = png_path
            if preprocess:
                pre_path = png_path.replace(".png", "_pre.png")
                if preprocess_image(png_path, pre_path):
                    target = pre_path
                else:
                    log.warning("Preprocessing failed for page %d; using original.", i)

            last_result = None
            for attempt in range(1, max_page_retries + 1):
                if mode is ExtractionMode.OCR:
                    result = ocr_image(target, reader.readtext)
                else:
                    result = document_ai_image(
                        target,
                        backend=backend,
                        ocr_api=reader.readtext if reader else None,
                        azure_client=azure_client,
                    )

                if _is_valid_extraction(result, mode):
                    if attempt > 1:
                        log.info("Page %d recovered on attempt %d/%d.", i, attempt, max_page_retries)
                    return result

                last_result = result
                if attempt < max_page_retries:
                    wait = min(2 ** (attempt - 1), 8)
                    log.warning(
                        "Page %d produced no text (attempt %d/%d); retrying in %ds …",
                        i, attempt, max_page_retries, wait,
                    )
                    time.sleep(wait)

            log.error("Page %d still invalid after %d attempts; giving up.", i, max_page_retries)
            return last_result

        page_results: list = [None] * n
        log.info("Extracting %d page(s) with %d worker(s) …", n, workers)

        if workers == 1:
            for idx, png_path in enumerate(png_paths):
                log.info("Extracting page %d / %d …", idx + 1, n)
                page_results[idx] = _extract_page(idx + 1, png_path)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(_extract_page, idx + 1, png_path): idx
                    for idx, png_path in enumerate(png_paths)
                }
                done = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    page_results[idx] = future.result()
                    done += 1
                    log.info("Extracted page %d / %d (page #%d done)", done, n, idx + 1)

    # -- Merge and write -----------------------------------------------------
    merged = (
        merge_ocr(page_results)
        if mode is ExtractionMode.OCR
        else merge_document_ai(page_results)
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)
    log.info("Output written to %s (%d chars)", output_path, len(merged))

    failed = sum(1 for r in page_results if not _is_valid_extraction(r, mode))
    if failed:
        log.warning("Finished with %d/%d page failure(s).", failed, len(page_results))
    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract text from a PDF via OCR or Document AI.")
    p.add_argument("--input",  help="Source PDF path.")
    p.add_argument("--output", help="Output text file path.")
    p.add_argument("--mode",    choices=["ocr", "document_ai"], default="document_ai")
    p.add_argument("--backend", choices=["easyocr", "azure"],   default="azure",
                   help="Document AI backend (only used with --mode document_ai).")
    p.add_argument("--preprocess", action="store_true", help="Adaptive threshold before OCR.")
    p.add_argument("--dpi",  type=int, default=300)
    p.add_argument("--languages", nargs="+", default=["ch_tra", "en"], metavar="LANG")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--workers", type=int, default=8,
                   help="Page-extraction concurrency (default: 8 for azure, 1 for easyocr).")
    p.add_argument("--page-retries", type=int, default=5,
                   help="Attempts per page before giving up (default: 5).")
    args = p.parse_args()

    success = run(
        pdf_path    = args.input,
        output_path = args.output,
        mode        = ExtractionMode.OCR if args.mode == "ocr" else ExtractionMode.DOCUMENT_AI,
        backend     = args.backend,
        languages   = args.languages,
        gpu         = not args.no_gpu,
        dpi         = args.dpi,
        preprocess  = args.preprocess,
        max_workers = args.workers,
        max_page_retries = args.page_retries,
    )
    raise SystemExit(0 if success else 1)