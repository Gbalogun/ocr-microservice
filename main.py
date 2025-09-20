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


def extract_fields(text: str) -> dict:
    # Vehicle registration (e.g., FP63VKN)
    vrm_match = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", text)

    # Dates (dd/mm/yyyy)
    date_matches = re.findall(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    contravention_date = date_matches[0] if date_matches else None
    due_date = date_matches[-1] if len(date_matches) > 1 else None

    # Contravention code (e.g., 07, 21A, etc.)
    contravention_code_match = re.search(r"\b\d{2}[A-Z]?\b", text)

    # Contravention type (reason for issue)
    contravention_type_match = None
    contravention_keywords = [
        "parking", "restricted", "ticket", "bay", "session",
        "payment", "disabled", "bus lane", "loading"
    ]
    for line in text.splitlines():
        if any(word in line.lower() for word in contravention_keywords):
            contravention_type_match = line.strip()
            break

    # Location (look for something resembling an address or site name)
    location_match = None
    for line in text.splitlines():
        if re.search(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", line) and re.search(r"\d{1,4}\s?[A-Z]{1,2}\d[A-Z]{2}", line):
            location_match = line.strip()
            break

    # Authority (often includes "Council", "Limited", "Ltd", "Borough")
    authority_match = None
    for line in text.splitlines():
        if any(keyword in line for keyword in ["Council", "Borough", "Limited", "Ltd"]):
            authority_match = line.strip()
            break

    # Fine amounts (£xx) — discounted and full
    fine_amounts = re.findall(r"£\s?(\d+(?:\.\d{2})?)", text)
    fine_amount_discounted = None
    fine_amount_full = None
    if fine_amounts:
        amounts = sorted(set(float(a) for a in fine_amounts))
        if len(amounts) >= 2:
            fine_amount_discounted, fine_amount_full = amounts[0], amounts[-1]
        elif len(amounts) == 1:
            fine_amount_full = amounts[0]

    return {
        "vrm": vrm_match.group(0) if vrm_match else None,
        "contravention_date": contravention_date,
        "contravention_code": contravention_code_match.group(0) if contravention_code_match else None,
        "contravention_type": contravention_type_match,
        "pcn_number": None,  # Can add regex later if needed
        "location": location_match,
        "authority": authority_match,
        "fine_amount_discounted": fine_amount_discounted,
        "fine_amount_full": fine_amount_full,
        "due_date": due_date,
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
