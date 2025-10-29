import io
import os
import re
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract


app = FastAPI()

# CORS (unchanged – allow your UI to call this service)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust if you need to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- CONFIG / ENV ----------
# If your Render build installed poppler-utils, this is typically /usr/bin
POPPLER_PATH = os.getenv("POPPLER_PATH")  # e.g. "/usr/bin" or None
# Limit pages for performance (change if you want full PDF)
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))
PDF_DPI = int(os.getenv("PDF_DPI", "200"))  # DPI for rasterization


@app.get("/")
def home():
    return {"status": "ok", "message": "OCR microservice is running"}

@app.get("/health")
def health():
    return {"ok": True}


# --------- YOUR EXTRACTION LOGIC (unchanged aside from being factored) ----------
CONTRAVENTION_TYPES = {
    "01": "Parking in a restricted area",
    "02": "Overstaying in parking bay",
    "03": "No valid parking ticket",
    "04": "Parking without payment",
    "05": "Parking without payment",
    "06": "Parking without payment",
    "12": "Parking in disabled bay",
    "16": "Loading in restricted hours",
    "23": "Bus lane violation",
}

def extract_fields(text: str) -> dict:
    """
    Your existing rule-based extractor. Kept as-is, only lightly structured.
    """
    lines = text.splitlines()
    data = {
        "pcn_number": None,
        "vrm": None,
        "contravention_date": None,
        "contravention_code": None,
        "contravention_type": None,
        "location": None,
        "authority": None,
        "fine_amount_discounted": None,
        "fine_amount_full": None,
        "due_date": None,
    }

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()

        # PCN reference / number
        if any(k in lower_line for k in ["pcn reference", "reference no", "ref", "reference number", "pcn number", "reference:"]):
            m = re.search(r"[A-Z0-9]{5,}", line)
            if m:
                data["pcn_number"] = m.group(0)

        # VRM like FP63VKN
        if any(k in lower_line for k in ["vehicle", "registration", "vrm", "plate"]):
            m = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", line)
            if m:
                data["vrm"] = m.group(0)

        # Dates: contravention/issue vs payment
        if any(k in lower_line for k in ["date", "issued", "issue", "offence", "contravention", "payment", "by"]):
            m = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
            if m:
                if "payment" in lower_line or "by" in lower_line:
                    # treat as final payment due date
                    data["due_date"] = m.group(0)
                else:
                    data["contravention_date"] = m.group(0)

        # Contravention code/type
        if any(k in lower_line for k in ["contravention", "reason", "offence"]):
            code_match = re.search(r"\b\d{2}[A-Z]?\b", line)
            if code_match:
                code = code_match.group(0)
                data["contravention_code"] = code
                data["contravention_type"] = CONTRAVENTION_TYPES.get(code)

        # Location
        if "location" in lower_line:
            # take everything after colon if present, else full line tail
            parts = line.split(":", 1)
            data["location"] = parts[1].strip() if len(parts) == 2 else line.strip()

        # Authority (council/borough/Ltd)
        if any(k in lower_line for k in ["authority", "council", "borough", "limited", "ltd", "parking company"]):
            data["authority"] = line.strip()

        # Fine amounts — pick discounted vs full
        if "£" in line:
            amounts = re.findall(r"£\s?(\d+(?:\.\d{2})?)", line)
            for a in amounts:
                try:
                    value = float(a)
                except ValueError:
                    continue
                if any(tok in lower_line for tok in ["discount", "within 14", "reduced"]):
                    data["fine_amount_discounted"] = value
                else:
                    # keep the largest as the "full" amount
                    current = data["fine_amount_full"] or 0
                    data["fine_amount_full"] = max(current, value)

    return data


# --------- NEW: SAFE PDF → IMAGE CONVERSION ----------
def convert_pdf_to_images(pdf_bytes: bytes, max_pages: int = MAX_PAGES, dpi: int = PDF_DPI) -> List[Image.Image]:
    """
    Convert a PDF to a list of PIL Images using pdf2image.
    Uses POPPLER_PATH env var if set (so pdf2image can find 'pdftoppm').
    Raises a RuntimeError with a clear message if conversion fails.
    """
    kwargs = {"dpi": dpi, "fmt": "jpeg"}  # fmt keeps memory lower and OCR happier
    if POPPLER_PATH:  # Let pdf2image know exactly where pdftoppm is
        kwargs["poppler_path"] = POPPLER_PATH

    try:
        pages = convert_from_bytes(pdf_bytes, **kwargs)
        if not pages:
            raise RuntimeError("PDF conversion produced 0 pages.")
        return pages[:max_pages]
    except Exception as e:
        raise RuntimeError(
            "PDF conversion failed. Ensure 'poppler-utils' is installed "
            f"and accessible. Details: {e}"
        )


def open_image_from_bytes(raw: bytes) -> Image.Image:
    """
    Open a raw image in a safe way (and force RGB to avoid pillow mode issues).
    """
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")


def ocr_images(images: List[Image.Image]) -> str:
    """
    OCR a list of PIL images and return concatenated text.
    """
    text_chunks: List[str] = []
    for img in images:
        # You can tune these configs if you know fonts/layouts
        txt = pytesseract.image_to_string(img)
        if txt:
            text_chunks.append(txt)
    return "\n".join(text_chunks)


# --------- OCR ENDPOINT (minimal change to your original shape) ----------
@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        raw = await file.read()

        # Reject PDFs for now
        if file.filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "PDF uploads are not yet supported. Please upload a clear image (JPEG or PNG)."
                },
            )

        # Proceed with image OCR
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as img_err:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid image format: {img_err}"}
            )

        text = pytesseract.image_to_string(image)
        extracted = extract_fields(text)

        return JSONResponse(content={"success": True, "data": extracted})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"OCR processing failed: {e}"}
        )
