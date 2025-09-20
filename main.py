from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import re
import io

app = FastAPI()

# Fix CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "OCR microservice is running"}


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
    # PCN number
    pcn_match = re.search(r"(PCN|Reference)\s*[:\-]?\s*([A-Z0-9]+)", text, re.IGNORECASE)

    # Contravention code
    contravention_code_match = re.search(r"\b\d{2}[A-Z]?\b", text)
    contravention_code = contravention_code_match.group(0) if contravention_code_match else None

    # Contravention type mapped
    contravention_type = CONTRAVENTION_TYPES.get(contravention_code, None)

    # Fine amounts with context
    fine_amount_discounted = None
    fine_amount_full = None
    for line in text.splitlines():
        if "discount" in line.lower():
            match = re.search(r"£\s?(\d+(?:\.\d{2})?)", line)
            if match:
                fine_amount_discounted = float(match.group(1))
        elif any(keyword in line.lower() for keyword in ["full", "amount due", "total"]):
            match = re.search(r"£\s?(\d+(?:\.\d{2})?)", line)
            if match:
                fine_amount_full = float(match.group(1))

    # If still missing, fallback to min/max
    if not (fine_amount_discounted and fine_amount_full):
        fine_amounts = re.findall(r"£\s?(\d+(?:\.\d{2})?)", text)
        if fine_amounts:
            amounts = sorted(set(float(a) for a in fine_amounts))
            if len(amounts) >= 2:
                fine_amount_discounted = fine_amount_discounted or amounts[0]
                fine_amount_full = fine_amount_full or amounts[-1]
            elif len(amounts) == 1:
                fine_amount_full = fine_amount_full or amounts[0]

    # Due date for payment
    due_date_match = re.search(r"BY\s+(\d{2}/\d{2}/\d{4})", text)
    due_date = due_date_match.group(1) if due_date_match else None

    return {
        "pcn_number": pcn_match.group(2) if pcn_match else None,
        "contravention_code": contravention_code,
        "contravention_type": contravention_type,
        "fine_amount_discounted": fine_amount_discounted,
        "fine_amount_full": fine_amount_full,
        "due_date": due_date,
        # ... (rest stays same: vrm, authority, location, etc.)
    }


@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(content)
        else:
            image = Image.open(io.BytesIO(content))
            images = [image]

        text = "".join(pytesseract.image_to_string(img) for img in images)
        extracted = extract_fields(text)
        return JSONResponse(content=extracted)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
