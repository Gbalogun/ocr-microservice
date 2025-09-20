from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import re
import io

app = FastAPI()

# Enable CORS (allow all origins for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "OCR microservice is running"}

# Map contravention codes to human-friendly descriptions
CONTRAVENTION_TYPES = {
    "01": "Parking in a restricted area",
    "02": "Overstaying in parking bay",
    "05": "No valid parking ticket",
    "06": "Parking without payment",
    "12": "Parking in disabled bay",
    "16": "Loading in restricted hours",
    "23": "Bus lane violation",
}

def extract_fields(text: str) -> dict:
    # Vehicle Reg Mark (VRM)
    vrm_match = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", text)

    # Contravention Date
    date_match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)

    # Contravention Code
    code_match = re.search(r"\b\d{2}[A-Z]?\b", text)
    contravention_code = code_match.group(0) if code_match else None
    contravention_type = CONTRAVENTION_TYPES.get(contravention_code, "Other" if contravention_code else None)

    # Fine Amount (£xx.xx)
    fine_match = re.search(r"£?\s?(\d+\.\d{2})", text)

    # PCN Number (very often alphanumeric, e.g. "PN12345678")
    pcn_match = re.search(r"\b[A-Z]{2}\d{6,8}\b", text)

    # Location (basic heuristic: look for "at <location>")
    location_match = re.search(r"at\s+([A-Za-z0-9\s,]+)", text, re.IGNORECASE)

    # Issuing Authority (look for "Issued by ..." or council keywords)
    authority_match = re.search(r"(Council|Authority|City of [A-Za-z]+)", text, re.IGNORECASE)

    return {
        "vrm": vrm_match.group(0) if vrm_match else None,
        "contravention_date": date_match.group(0) if date_match else None,
        "contravention_code": contravention_code,
        "contravention_type": contravention_type,
        "fine_amount": float(fine_match.group(1)) if fine_match else None,
        "pcn_number": pcn_match.group(0) if pcn_match else None,
        "location": location_match.group(1).strip() if location_match else None,
        "authority": authority_match.group(0) if authority_match else None,
    }

@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Convert PDF to images, or open image directly
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(content)
        else:
            image = Image.open(io.BytesIO(content))
            images = [image]

        # Run OCR
        text = "".join(pytesseract.image_to_string(img) for img in images)

        # Extract structured fields
        extracted = extract_fields(text)
        return JSONResponse(content=extracted)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
