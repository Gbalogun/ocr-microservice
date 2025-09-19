from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import re
import io

app = FastAPI(
    title="OCR Microservice",
    version="1.0.0",
    description="Extracts PCN details from images and PDFs"
)

# ✅ Enable CORS so Swagger + frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_fields(text: str) -> dict:
    # Basic regex extraction logic (can be improved later)
    vrm_match = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", text)
    date_match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    contravention_code_match = re.search(r"\b\d{2}[A-Z]?\b", text)

    return {
        "vrm": vrm_match.group(0) if vrm_match else None,
        "contravention_date": date_match.group(0) if date_match else None,
        "location": None,      # TODO: improve with NLP/geolocation
        "authority": None,     # TODO: match against known authorities
        "contravention_code": contravention_code_match.group(0) if contravention_code_match else None,
    }


@app.get("/")
async def home():
    return {"status": "ok", "message": "OCR microservice is running"}


@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(content)
        else:
            image = Image.open(io.BytesIO(content))
            images = [image]

        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)

        extracted = extract_fields(text)
        return JSONResponse(content=extracted)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
