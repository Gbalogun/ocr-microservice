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

    for i, line in enumerate(lines):
        lower_line = line.lower()

        # 🔹 PCN Number / Reference
        if any(k in lower_line for k in ["pcn", "reference", "ref","refernce number", "number"]):
            match = re.search(r"[A-Z0-9]{6,}", line)
            if match:
                data["pcn_number"] = match.group(0)

        # 🔹 Vehicle Registration
        if any(k in lower_line for k in ["vehicle", "registration", "vrm", "plate"]):
            match = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", line)
            if match:
                data["vrm"] = match.group(0)

        # 🔹 Dates (issue/contravention/due)
        if any(k in lower_line for k in ["date", "issue", "offence", "contravention", "payment", "by"]):
            match = re.search(r"\d{2}/\d{2}/\d{4}", line)
            if match:
                if "payment" in lower_line or "by" in lower_line:
                    data["due_date"] = match.group(0)
                else:
                    data["contravention_date"] = match.group(0)

        # 🔹 Contravention
        if any(k in lower_line for k in ["contravention", "reason", "offence"]):
            code_match = re.search(r"\b\d{2}[A-Z]?\b", line)
            if code_match:
                code = code_match.group(0)
                data["contravention_code"] = code
                data["contravention_type"] = CONTRAVENTION_TYPES.get(code, line.strip())

        # 🔹 Location
        if any(k in lower_line for k in ["location", "place", "street", "road", "car park"]):
            data["location"] = line.split(":")[-1].strip()

        # 🔹 Authority
        if any(k in lower_line for k in ["authority", "council", "borough", "ltd", "company"]):
            data["authority"] = line.strip()

        # 🔹 Fine Amounts
        if "£" in line:
            matches = re.findall(r"£\s?(\d+(?:\.\d{2})?)", line)
            for amt in matches:
                amount = float(amt)
                if "discount" in lower_line or "within 14" in lower_line:
                    data["fine_amount_discounted"] = amount
                else:
                    data["fine_amount_full"] = max(data["fine_amount_full"] or 0, amount)

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
