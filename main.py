from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import re
import io
from datetime import datetime 

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

def validate_date(date_str: str) -> str | None:
    """Ensure date is valid UK format DD/MM/YYYY, otherwise return None."""
    try:
        d = datetime.strptime(date_str, "%d/%m/%Y")
        return d.strftime("%d/%m/%Y")
    except Exception:
        return None

def extract_fields(text: str) -> dict:
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
        "discount_deadline": None,
        "due_date": None,
    }

    amounts_found = []

    for i, line in enumerate(lines):
        lower_line = line.lower()

        # 🔹 PCN Number / Reference
        if any(k in lower_line for k in ["pcn reference", "reference no.", "ref", "reference number"]):
            match = re.search(r"[A-Z0-9]{6,}", line)
            if match:
                data["pcn_number"] = match.group(0)

        # 🔹 Vehicle Registration
        if any(k in lower_line for k in ["vehicle", "registration", "vrm", "plate"]):
            match = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", line.replace(" ", ""))
            if match:
                data["vrm"] = match.group(0)

        # 🔹 Dates (issue/contravention/due/discount)
        if any(k in lower_line for k in ["date", "issue", "offence", "contravention", "payment", "by", "discount"]):
            matches = re.findall(r"\d{2}/\d{2}/\d{4}", line)
            for m in matches:
                m_valid = validate_date(m)
                if m_valid:
                    if "payment" in lower_line or "by" in lower_line:
                        data["due_date"] = m_valid
                    elif "discount" in lower_line or "within 14" in lower_line:
                        data["discount_deadline"] = m_valid
                    else:
                        data["contravention_date"] = m_valid

        # 🔹 Contravention
        if any(k in lower_line for k in ["contravention", "reason", "offence"]):
            # Fix OCR splitting like "0 3"
            code_match = re.search(r"\b(\d\s?\d[A-Z]?)\b", line)
            if code_match:
                code = code_match.group(1).replace(" ", "")
                data["contravention_code"] = code
                data["contravention_type"] = CONTRAVENTION_TYPES.get(code, line.strip())

        # 🔹 Location
        if "location" in lower_line:
            data["location"] = line.split(":")[-1].strip()

        # 🔹 Authority
        if any(k in lower_line for k in ["authority", "council", "borough", "ltd", "limited"]):
            data["authority"] = line.strip()

        # 🔹 Fine Amounts
        if "£" in line:
            matches = re.findall(r"£\s?(\d+(?:\.\d{2})?)", line)
            for amt in matches:
                amount = float(amt)
                amounts_found.append(amount)
                if "discount" in lower_line or "within 14" in lower_line:
                    data["fine_amount_discounted"] = amount

    # Post-process fine amounts
    if amounts_found:
        data["fine_amount_full"] = max(amounts_found)
        if not data["fine_amount_discounted"] and len(amounts_found) > 1:
            data["fine_amount_discounted"] = min(amounts_found)

    return data


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
