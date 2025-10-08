import os, sys, pdfplumber, pytesseract
from PIL import Image

print("=== Debug PDF Check ===")
print("Python executable:", sys.executable)
pdf_path = "data/input/newspaper.pdf"
print("PDF exists:", os.path.exists(pdf_path))

if not os.path.exists(pdf_path):
    print("❌ PDF not found. Place it in data/input/newspaper.pdf")
    sys.exit()

with pdfplumber.open(pdf_path) as pdf:
    print("✅ PDF opened successfully")
    print("Number of pages:", len(pdf.pages))
    first_page = pdf.pages[0]
    text = first_page.extract_text()
    print("Selectable text present?", bool(text and text.strip()))
    if not text:
        print("🧾 Running OCR on page 1…")
        img = first_page.to_image(resolution=300).original
        tpath = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
        print("Using Tesseract path:", tpath)
        ocr_text = pytesseract.image_to_string(img, lang="ara+fra")
        print("OCR text length:", len(ocr_text))
        print("OCR sample:", ocr_text[:400])
    else:
        print("✅ Text extraction succeeded!")
        print("Text sample:", text[:400])
