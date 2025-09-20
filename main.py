from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import re
import io

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def home():
    return {"status": "ok", "message": "OCR microservice is running"}

# Common contravention code → description mapping (extend as needed)
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

# Regex helpers
MONEY = re.compile(r"£?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2})")
DATE_DMY = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
VRM = re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{3}\b")
CODE = re.compile(r"\b\d{2}[A-Z]?\b")

# “Location: …” or “at … Road/Street/…”
LOCATION = re.compile(
    r"(?:Location[:\s]+(?P<loc1>.+?)\s*$)|(?:\bat\s+(?P<loc2>[^,\n]{0,80}?(?:Road|Rd|Street|St|Avenue|Ave|Lane|Ln|Way|Close|Cl|Drive|Dr|Crescent|Cres|Terrace|Terr|Place|Pl|Park|Car Park)\b[^,\n]*) )",
    re.IGNORECASE | re.MULTILINE,
)

# Council OR company authorities
AUTHORITY = re.compile(
    r"(?:(?:[A-Z][A-Za-z& ]+?\s(?:Council|Borough|City Council|Parking Services|Parking Services Ltd|Parking Services Limited))|"
    r"(?:[A-Z][A-Za-z& ]+?\s(?:Ltd|Limited|PLC|LLP|Group|Holdings|Company)))",
    re.IGNORECASE,
)

# “PCN number”, “Penalty Charge Notice number”, “Reference”
PCN_NO = re.compile(
    r"(?:PCN|Penalty\s*Charge\s*Notice)\s*(?:No\.?|Number|Ref(?:erence)?)[\s:]*([A-Z0-9\-]{6,14})",
    re.IGNORECASE,
)

# Direct “reason” phrasing
REASON = re.compile(
    r"(?:Contravention(?:\s*reason)?|Reason\s*(?:for\s*issue)?)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)

def _norm_money(m: str) -> float:
    return float(m.replace("£", "").replace(",", "").strip())

def _window(text: str, center_idx: int, radius: int = 100) -> str:
    start = max(0, center_idx - radius)
    end = min(len(text), center_idx + radius)
    return text[start:end]

def extract_fields(text: str) -> dict:
    # Normalised copy for keyword proximity checks
    low = text.lower()

    # --- VRM ---
    vrm = None
    m = VRM.search(text)
    if m:
        vrm = m.group(0)

    # --- Date (prefer “contravention date” if present) ---
    contravention_date = None
    m_pref = re.search(
        r"(?:date\s*(?:of)?\s*contravention|contravention\s*date)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        low,
        re.IGNORECASE,
    )
    if m_pref:
        contravention_date = m_pref.group(1)
    else:
        m = DATE_DMY.search(text)
        if m:
            contravention_date = m.group(1)

    # --- Contravention code + type ---
    contravention_code = None
    contravention_type = None
    m = CODE.search(text)
    if m:
        contravention_code = m.group(0)
        contravention_type = CONTRAVENTION_TYPES.get(contravention_code, "Other")

    # Try to get an explicit reason string if available
    contravention_reason = None
    for line in text.splitlines():
        rm = REASON.search(line)
        if rm:
            # stop at end of line or first dot/bracket
            contravention_reason = re.split(r"[.\(\[]", rm.group(1).strip())[0].strip()
            break
    if contravention_reason and contravention_reason.lower() != "other":
        contravention_type = contravention_reason

    # --- Money: discounted + full ---
    # Strategy: collect all amounts; classify by nearby keywords
    amounts = []
    for m in MONEY.finditer(text):
        val = _norm_money(m.group(1))
        ctx = _window(low, m.start())
        amounts.append((val, ctx))

    fine_amount_discounted = None
    fine_amount_full = None
    if amounts:
        # Try to classify by keywords
        for val, ctx in amounts:
            if any(k in ctx for k in ["discount", "reduced", "within 14", "within fourteen", "if paid within", "early payment"]):
                fine_amount_discounted = min(val, fine_amount_discounted or val)
            if any(k in ctx for k in ["full amount", "after 14", "after fourteen", "balance", "standard charge", "if not paid"]):
                fine_amount_full = max(val, fine_amount_full or val)

        # If classification didn’t work but we have ≥2 amounts, pick smallest as discounted, largest as full
        if (fine_amount_discounted is None or fine_amount_full is None) and len(amounts) >= 2:
            vals = sorted(v for v, _ in amounts)
            if fine_amount_discounted is None:
                fine_amount_discounted = vals[0]
            if fine_amount_full is None:
                fine_amount_full = vals[-1]

        # If only one amount, treat it as full
        if len(amounts) == 1 and fine_amount_full is None:
            fine_amount_full = amounts[0][0]

    # --- PCN number ---
    pcn_number = None
    m = PCN_NO.search(text)
    if m:
        pcn_number = m.group(1).strip("-")

    # --- Location ---
    location = None
    lm = LOCATION.search(text)
    if lm:
        location = (lm.group("loc1") or lm.group("loc2") or "").strip(" :-").strip()
        # Avoid capturing very long/wrong chunks
        if len(location) > 120:
            location = None

    # --- Authority (council OR company) ---
    authority = None
    # Prefer explicit “Issued by …” style
    am = re.search(r"(?:Issued\s*by|Enforcement\s*Authority)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if am:
        authority = am.group(1).strip()
        # truncate at line end or after common terminators
        authority = re.split(r"[.\n\r]", authority)[0].strip()

    if not authority:
        am = AUTHORITY.search(text)
        if am:
            authority = am.group(0).strip()
    if authority:
        # Normalise spacing/casing a bit
        authority = re.sub(r"\s{2,}", " ", authority).strip()

    return {
        "vrm": vrm,
        "contravention_date": contravention_date,
        "contravention_code": contravention_code,
        "contravention_type": contravention_type,
        "pcn_number": pcn_number,
        "location": location,
        "authority": authority,
        "fine_amount_discounted": fine_amount_discounted,
        "fine_amount_full": fine_amount_full,
    }

@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Convert PDF → images or open image directly
        if file.filename.lower().endswith(".pdf"):
            # Slightly higher DPI for better OCR; adjust if needed
            images = convert_from_bytes(content, dpi=200)
        else:
            image = Image.open(io.BytesIO(content))
            images = [image]

        # OCR (psm 6 handles blocks of text reasonably well)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, config="--psm 6")

        extracted = extract_fields(text)
        return JSONResponse(content=extracted)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
