# ===============================
# 🚀 Hybrid OCR Microservice (OpenAI + Tesseract)
# ===============================

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image
import pytesseract
import io, os, base64, re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ✅ Allow CORS for frontend access
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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# Helper Function: Field Extraction (for Tesseract fallback)
# -------------------------------
def extract_fields(text: str) -> dict:
    """Extract common PCN fields from OCR text."""
    lines = text.splitlines()
    data = {
        "pcn_number": None,
        "vrm": None,
        "contravention_date": None,
        "due_date": None,
        "fine_amount_full": None,
        "fine_amount_discounted": None,
        "location": None,
        "authority": None,
    }

    for line in lines:
        lower_line = line.lower().strip()
        if not lower_line:
            continue

        # Reference Number / PCN
        if "reference" in lower_line or "pcn" in lower_line:
            m = re.search(r"[A-Z0-9]{6,}", line)
            if m:
                data["pcn_number"] = m.group(0)

        # VRM / Vehicle Registration
        if "vehicle registration" in lower_line or "reg" in lower_line or "vrm" in lower_line:
            m = re.search(r"\b[A-Z]{2}[0-9]{2}[A-Z]{3}\b", line)
            if m:
                data["vrm"] = m.group(0)

        # Dates (Contravention, Due)
        if "date" in lower_line:
            m = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
            if m:
                if "due" in lower_line:
                    data["due_date"] = m.group(0)
                else:
                    data["contravention_date"] = m.group(0)

        # Fine Amount
        if "amount" in lower_line or "charge" in lower_line:
            m = re.findall(r"£\s?(\d+(?:\.\d{2})?)", line)
            if m:
                values = [float(x) for x in m]
                data["fine_amount_full"] = max(values)
                if len(values) > 1:
                    data["fine_amount_discounted"] = min(values)

        # Location
        if "location" in lower_line:
            data["location"] = line.split(":", 1)[-1].strip()

        # Authority
        if "council" in lower_line or "limited" in lower_line or "parkmaven" in lower_line:
            data["authority"] = "ParkMaven Limited"

    return data

# -------------------------------
# Main OCR Endpoint
# -------------------------------
@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """Extracts key fields (VRM, fine, location, etc.) from uploaded PCN."""
    image_bytes = await file.read()

    # Convert image to base64 for OpenAI
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    try:
        # STEP 1: Try OpenAI Vision OCR
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR parser that extracts structured data from PCN (Penalty Charge Notice) images. "
                        "Return ONLY valid JSON with the following fields: "
                        "{VRM, ReferenceNumber, FineAmount, DiscountedAmount, DateOfEvent, PaymentDue, Location, Authority}."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract data from this PCN image and respond only in JSON format."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded}"}
                    ]
                }
            ],
            temperature=0
        )

        text_output = response.choices[0].message.content

        return JSONResponse(
            content={
                "success": True,
                "source": "openai",
                "data": text_output
            }
        )

    except Exception as e:
        # STEP 2: Fallback to Tesseract OCR
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            text = pytesseract.image_to_string(image, config="--psm 6 --oem 3")
            extracted = extract_fields(text)

            return JSONResponse(
                content={
                    "success": True,
                    "source": "tesseract",
                    "data": extracted,
                    "error": str(e)
                }
            )
        except Exception as err:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"OCR failed: {str(err)}"
                },
                status_code=500
            )

# -------------------------------
# Health Check Endpoint
# -------------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "Hybrid OCR (OpenAI + Tesseract)"}
