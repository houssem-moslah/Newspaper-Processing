"""
pdf_pipeline.py
Step 1: PDF -> Structured Pages (jsonl)

Usage:
  python src/pdf_pipeline.py --pdf data/input/newspaper.pdf --out data/output/pages.jsonl

Features:
- Uses pdfplumber to extract selectable text
- Falls back to Tesseract OCR for scanned/image-only pages
- Splits page text into paragraphs and detects language per paragraph
- Produces one JSON object per page written as JSON Lines
- Verbose logging with timestamps, runtime and memory usage (psutil optional)
"""

import argparse
import pdfplumber
import pytesseract
from PIL import Image
import io
import json
import datetime
import os
import time
from dateutil import tz
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm
import logging

# optional psutil for memory usage
try:
    import psutil
except Exception:
    psutil = None

DetectorFactory.seed = 0  # deterministic language detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("pdf_pipeline")

# Tesseract path: prefer env var TESSERACT_CMD, fallback to common install path.
DEFAULT_TESSERACT_PATHS = [
    os.environ.get("TESSERACT_CMD"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

for p in DEFAULT_TESSERACT_PATHS:
    if p and os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break

if not getattr(pytesseract.pytesseract, "tesseract_cmd", None):
    logger.warning(
        "Tesseract executable not found. If OCR is needed, set env var TESSERACT_CMD "
        "or install Tesseract and update the path."
    )


def safe_detect_language(text: str) -> str:
    """
    Returns 'ar' for Arabic, 'fr' for French, or 'unknown'.
    Uses langdetect per-paragraph; small texts may be inconclusive.
    """
    if not text or not text.strip():
        return "unknown"
    try:
        lang = detect(text)
    except LangDetectException:
        return "unknown"
    # langdetect returns codes like 'ar', 'fr', 'en', etc.
    if lang.startswith("ar"):
        return "ar"
    if lang.startswith("fr"):
        return "fr"
    return "unknown"


def ocr_image_from_page(page, ocr_langs="ara+fra"):
    """
    Convert pdfplumber page image to PIL.Image and run pytesseract.
    Returns OCR'ed text.
    """
    # Use a high resolution image from pdfplumber to improve OCR
    img = page.to_image(resolution=300).original  # returns PIL.Image
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    try:
        text = pytesseract.image_to_string(img, lang=ocr_langs)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        text = ""
    return text


def split_paragraphs(text: str):
    """
    Split a page text into paragraphs. Strategy:
    - Normalize line endings
    - Split on two or more newlines
    - Fallback: split on single newline and join short lines into paragraphs
    """
    if not text:
        return []

    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
    if parts:
        return parts

    # fallback: group lines into paragraphs by blank-line-like heuristics
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    paragraphs = []
    if not lines:
        return []
    current = lines[0]
    for ln in lines[1:]:
        # if line ends with a punctuation likely end of paragraph
        if current.endswith((".", "?", "!", "؟", ".", "»", "»")):
            paragraphs.append(current)
            current = ln
        else:
            # join lines that are likely wrapped
            current = current + " " + ln
    if current:
        paragraphs.append(current)
    return [p.strip() for p in paragraphs if p.strip()]


def process_pdf(pdf_path: str, out_path: str, ocr_langs="ara+fra", force_ocr=False):
    """
    Main function:
    - iterate pages
    - extract selectable text via pdfplumber
    - if text is empty or force_ocr: run OCR
    - split into paragraphs, detect language per paragraph and aggregate into page fields
    - write pages as JSON lines to out_path
    """
    start_all = time.time()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pages_output = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"Opened PDF {pdf_path} ({total_pages} pages)")

        for i, page in enumerate(tqdm(pdf.pages, desc="Pages"), start=1):
            t0 = time.time()
            logger.info(f"Processing page {i}/{total_pages}")
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.exception(f"pdfplumber.extract_text failed on page {i}: {e}")
                text = ""

            used_ocr = False
            # If no text or explicitly forcing OCR -> OCR the page
            if not text.strip() or force_ocr:
                logger.debug(f"Running OCR for page {i}")
                text = ocr_image_from_page(page, ocr_langs=ocr_langs)
                used_ocr = True

            paragraphs = split_paragraphs(text)
            # If still empty, try OCR again with fallback
            if not paragraphs and not used_ocr:
                logger.debug(f"No paragraphs found; attempting OCR fallback for page {i}")
                text = ocr_image_from_page(page, ocr_langs=ocr_langs)
                paragraphs = split_paragraphs(text)
                used_ocr = True

            # Build per-language concatenations
            fr_parts = []
            ar_parts = []
            unknown_parts = []
            for para in paragraphs:
                lang = safe_detect_language(para)
                if lang == "fr":
                    fr_parts.append(para)
                elif lang == "ar":
                    ar_parts.append(para)
                else:
                    # Some paragraphs are mixed or short; put into unknown for now
                    unknown_parts.append(para)

            # Heuristic: if unknown parts exist, attempt to check by char set
            for u in list(unknown_parts):
                # if contains Arabic characters, assign to ar
                if any("\u0600" <= ch <= "\u06FF" for ch in u):
                    ar_parts.append(u)
                    unknown_parts.remove(u)
                else:
                    # otherwise append to French (project expects French as main NER source)
                    fr_parts.append(u)
                    unknown_parts.remove(u)

            page_content_markdown = "\n\n".join(paragraphs).strip()
            page_content_fr = "\n\n".join(fr_parts).strip()
            page_content_arabic = "\n\n".join(ar_parts).strip()

            processed_at = datetime.datetime.now(tz=tz.tzlocal()).isoformat()

            page_entry = {
                "page_num": i,
                "page_content_markdown": page_content_markdown,
                "page_content_fr": page_content_fr,
                "page_content_arabic": page_content_arabic,
                "processed_at": processed_at,
            }

            pages_output.append(page_entry)

            duration = time.time() - t0
            mem = None
            if psutil:
                try:
                    p = psutil.Process(os.getpid())
                    mem = p.memory_info().rss / (1024 * 1024)  # MB
                except Exception:
                    mem = None

            logger.info(
                f"Processed page {i} (ocr={used_ocr}) duration={duration:.2f}s"
                + (f", mem={mem:.1f}MB" if mem else "")
            )

    # Write JSON Lines
    with open(out_path, "w", encoding="utf-8") as fh:
        for page in pages_output:
            fh.write(json.dumps(page, ensure_ascii=False) + "\n")

    total_time = time.time() - start_all
    logger.info(f"Saved {len(pages_output)} pages to {out_path}")
    logger.info(f"Total time: {total_time:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="PDF -> structured pages (jsonl)")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", default="data/output/pages.jsonl", help="Output JSONL path")
    parser.add_argument("--ocr-langs", default="ara+fra", help="Tesseract languages (e.g. ara+fra)")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR for all pages")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_pdf(args.pdf, args.out, ocr_langs=args.ocr_langs, force_ocr=args.force_ocr)
