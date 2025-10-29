import io
import re
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract


app = FastAPI(
    title="Auto Nominate OCR Microservice",
    description="Extracts PCN details (VRM, fine, date, authority, etc.) from uploaded images.",
    version="1.0.0"
)

# CORS (allow Softgen frontend or other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────────
# FIELD EXTRACTION LOGIC
# ───────────────────────────────
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
    """Extract relevant PCN fields from OCR text."""
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

        # PCN Number
        if any(k in lower_line for k in ["pcn reference", "reference", "pcn number", "ref"]):
            m = re.search(r"[A-Z0-9]{5,}", line)
            if m:
                data["pcn_number"] = m.group(0)

        # VRM (vehicle registration mark)
        if re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", line):
            m = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", line)
            if m:
                data["vrm"] = m.group(0)

        # Date
        if any(k in lower_line for k in ["date", "issued", "issue", "offence", "contravention", "payment", "by"]):
            m = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
            if m:
                if "payment" in lower_line or "by" in lower_line:
                    data["due_date"] = m.group(0)
                else:
                    data["contravention_date"] = m.group(0)

        # Contravention code
        if any(k in lower_line for k in ["contravention", "reason", "offence"]):
            m = re.search(r"\b\d{2}[A-Z]?\b", line)
            if m:
                code = m.group(0)
                data["contravention_code"] = code
                data["contravention_type"] = CONTRAVENTION_TYPES.get(code)

        # Location
        if "location" in lower_line:
            parts = line.split(":", 1)
            data["location"] = parts[1].strip() if len(parts) == 2 else line.strip()

        # Authority
        if any(k in lower_line for k in ["authority", "council", "borough", "ltd", "limited", "parking company"]):
            data["authority"] = line.strip()

        # Fine amounts
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
                    current = data["fine_amount_full"] or 0
                    data["fine_amount_full"] = max(current, value)

    return data


# ───────────────────────────────
# ROUTES
# ───────────────────────────────

@app.get("/")
def home():
    return {"status": "ok", "message": "OCR microservice is running."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """
    Extracts text data from an uploaded image (JPEG, PNG).
    Rejects PDFs (for future support).
    """
    try:
        # Check file type
        if file.filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "PDF uploads are not yet supported. Please upload a clear image (JPEG or PNG)."
                }
            )

        # Load image safely
        raw = await file.read()
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as img_err:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid image format: {img_err}"}
            )

        # Run OCR
        text = pytesseract.image_to_string(image)

        # Extract structured fields
        extracted = extract_fields(text)

        # Return results
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "OCR extraction successful",
                "data": extracted
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"OCR processing failed: {e}"
            }
        )
