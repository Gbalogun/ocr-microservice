from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import io, os, re

from utils_openai import extract_with_openai

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "OCR microservice is running", "mode": os.getenv("OCR_MODE", "tesseract")}

# -------------------------------------------------------------------
# Fallback regex-based extraction (if using Tesseract)
# -------------------------------------------------------------------
def extract_fields_tesseract(text: str) -> dict:
    vrm_match = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", text)
    date_match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    contravention_code_match = re.search(r"\b\d{2}[A-Z]?\b", text)
    fine_amounts = re.findall(r"£\s?\d+(?:\.\d{2})?", text)
    discounted, full = None, None
    if fine_amounts:
        if len(fine_amounts) == 1:
            full = fine_amounts[0]
        else:
            discounted, full = sorted(fine_amounts, key=lambda x: float(x.replace("£", "")))

    return {
        "vrm": vrm_match.group(0) if vrm_match else None,
        "contravention_date": date_match.group(0) if date_match else None,
        "contravention_code": contravention_code_match.group(0) if contravention_code_match else None,
        "fine_amount_discounted": discounted,
        "fine_amount_full": full,
    }

# -------------------------------------------------------------------
# Main /ocr route
# -------------------------------------------------------------------
@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()
        mode = os.getenv("OCR_MODE", "tesseract").lower()

        # If using OpenAI (preferred)
        if mode == "openai":
            return await extract_with_openai(content, file.filename)

        # Default to Tesseract OCR
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(content)
        else:
            images = [Image.open(io.BytesIO(content))]

        text = "".join(pytesseract.image_to_string(img) for img in images)
        extracted = extract_fields_tesseract(text)

        return JSONResponse(content={"extracted": extracted, "engine": "tesseract"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
