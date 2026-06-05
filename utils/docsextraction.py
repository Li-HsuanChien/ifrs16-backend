"""
docsextraction.py

Pipeline for extracting text from PDF files via OCR or Document AI.

Stages:
    1. PDF  → per-page PNG images  (pdf2image)
    2. PNG  → text                 (EasyOCR  |  Document AI stub)
    3. Text → single output file   (flat concat for OCR, structured for Document AI)

Usage (CLI):
    python docsextraction.py input.pdf output.txt --mode ocr --preprocess --dpi 300

Usage (library):
    from docsextraction import run, ExtractionMode

    run("input.pdf", "output.txt", mode=ExtractionMode.OCR, preprocess=True)
"""

import argparse
import logging
import os
import re
import tempfile
from enum import Enum, auto
from typing import Callable, Optional

import cv2
import easyocr
from pdf2image import convert_from_path

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
def rasterise(pdf_path: str, tmpdir: str, dpi: int = 300) -> list[str]:
    """Convert each PDF page to a PNG inside tmpdir; return sorted PNG paths."""
    log.info("Rasterising %s at %d DPI …", pdf_path, dpi)
    pages = convert_from_path(pdf_path, dpi=dpi)
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
# Stage 2a – OCR extraction
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
# Stage 2b – Document AI extraction (stub)
# ---------------------------------------------------------------------------
def document_ai_image(image_path: str, ocr_api: OcrApiFunc) -> Optional[dict]:
    """
    Return structured extraction from one image, or None on failure.

    Replace the body with your real Document AI SDK call.
    Contract: return {"blocks": [{"text": str, "bbox": list, "confidence": float}]}
    """
    try:
        blocks = []
        for bbox, text, confidence in ocr_api(image_path):
            cleaned = re.sub(r'[a-zA-Z.!\'"^@#%/\-{}\[\]]', "", text)
            if cleaned.strip():
                blocks.append({"text": cleaned, "bbox": bbox, "confidence": confidence})
        log.info("Document AI: %s → %d blocks", image_path, len(blocks))
        return {"blocks": blocks}
    except Exception:
        log.exception("Document AI failed for %s", image_path)
        return None


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
    mode:        ExtractionMode = ExtractionMode.OCR,
    languages:   list[str]      = None,
    gpu:         bool           = True,
    dpi:         int            = 300,
    preprocess:  bool           = False,
) -> bool:
    """
    Full pipeline: PDF → PNGs → text → output file.

    Returns True if all pages extracted cleanly, False if any page failed
    (output file is still written with placeholders for bad pages).
    """
    log.info("Loading EasyOCR (languages=%s, gpu=%s) …", languages or ["ch_tra", "en"], gpu)
    reader = easyocr.Reader(languages or ["ch_tra", "en"], gpu=gpu)

    with tempfile.TemporaryDirectory(prefix="pdf_extract_") as tmpdir:
        png_paths = rasterise(pdf_path, tmpdir, dpi=dpi)
        if not png_paths:
            log.error("No pages found in %s", pdf_path)
            return False

        page_results = []
        for i, png_path in enumerate(png_paths, start=1):
            log.info("Extracting page %d / %d …", i, len(png_paths))

            target = png_path
            if preprocess:
                pre_path = png_path.replace(".png", "_pre.png")
                if preprocess_image(png_path, pre_path):
                    target = pre_path
                else:
                    log.warning("Preprocessing failed for page %d; using original.", i)

            if mode is ExtractionMode.OCR:
                page_results.append(ocr_image(target, reader.readtext))
            else:
                page_results.append(document_ai_image(target, reader.readtext))

    merged = merge_ocr(page_results) if mode is ExtractionMode.OCR else merge_document_ai(page_results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)
    log.info("Output written to %s (%d chars)", output_path, len(merged))

    failed = sum(1 for r in page_results if r is None)
    if failed:
        log.warning("Finished with %d/%d page failure(s).", failed, len(page_results))
    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract text from a PDF via OCR or Document AI.")
    p.add_argument("input",  help="Source PDF path.")
    p.add_argument("output", help="Output text file path.")
    p.add_argument("--mode", choices=["ocr", "document_ai"], default="ocr")
    p.add_argument("--preprocess", action="store_true", help="Adaptive threshold before OCR.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--languages", nargs="+", default=["ch_tra", "en"], metavar="LANG")
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()

    success = run(
        pdf_path    = args.input,
        output_path = args.output,
        mode        = ExtractionMode.OCR if args.mode == "ocr" else ExtractionMode.DOCUMENT_AI,
        languages   = args.languages,
        gpu         = not args.no_gpu,
        dpi         = args.dpi,
        preprocess  = args.preprocess,
    )
    raise SystemExit(0 if success else 1)