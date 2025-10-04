# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import re
import io
import math
from datetime import datetime
from typing import Optional, Tuple, List

# Optional OpenCV (better preprocessing). Falls back to PIL if not installed.
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    HAS_CV = True
except Exception:
    HAS_CV = False

app = FastAPI(title="OCR Microservice")

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


# ------------------------- CONFIG & HELPERS -------------------------

TESS_CONFIG = r'--oem 3 --psm 6 -l eng'

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

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def normalise_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())


def to_uk_date(s: str) -> Optional[str]:
    """Return DD/MM/YYYY when possible from a variety of formats."""
    s = s.strip()
    # 1) DD/MM/YYYY
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", s)
    if m:
        d, mth, y = m.groups()
        try:
            dt = datetime(int(y), int(mth), int(d))
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    # 2) D/M/YYYY (single-digit day or month)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s)
    if m:
        d, mth, y = m.groups()
        try:
            dt = datetime(int(y), int(mth), int(d))
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    # 3) "12 April 2025" style
    m = re.search(
        r"\b(\d{1,2})[^\w]?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s*(\d{4})\b",
        s, re.IGNORECASE)
    if m:
        d, mon, y = m.groups()
        mon_num = MONTHS.get(mon.lower())
        try:
            dt = datetime(int(y), mon_num, int(d))
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    return None


def vrm_candidates(text: str) -> List[str]:
    """
    Try to find UK-style VRMs. Prioritize those near VRM labels later.
    This regex is intentionally permissive; we'll post-filter.
    """
    patterns = [
        r"\b[A-Z]{2}\s?\d{2}\s?[A-Z]{3}\b",      # current style AB12 CDE
        r"\b[A-Z]{1,3}\s?\d{1,4}\b",            # older styles
        r"\b[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}\b"
    ]
    cands = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            cands.add(m.group(0).replace(" ", "").upper())
    # Filter out likely noise
    filtered = []
    for c in cands:
        if 4 <= len(c) <= 8 and not c.startswith("PCN"):
            filtered.append(c)
    return filtered


def score_near_label(text: str, label_keywords: List[str], value: str) -> int:
    """
    Simple proximity score: +1 if value appears within N chars of any label keyword.
    """
    t = text.lower()
    pos_val = t.find(value.lower())
    best = 0
    if pos_val == -1:
        return best
    for kw in label_keywords:
        pos_kw = t.find(kw)
        if pos_kw != -1:
            dist = abs(pos_kw - pos_val)
            if dist < 120:  # within ~120 chars = likely same block/line cluster
                best = max(best, 1)
    return best


def parse_amounts(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (discounted, full) based on £ amounts, smallest as discounted if text mentions discount."""
    amounts = [float(m.replace(",", "")) for m in re.findall(r"£\s*([\d,]+(?:\.\d{2})?)", text)]
    if not amounts:
        return None, None
    amounts = sorted(set(amounts))
    # heuristics
    discounted, full = None, None
    has_discount_words = re.search(r"discount|within\s+14|reduced", text, re.IGNORECASE)

    if has_discount_words and len(amounts) >= 2:
        discounted = amounts[0]
        full = amounts[-1]
    elif len(amounts) >= 2:
        # pick smallest as discounted, largest as full
        discounted = amounts[0]
        full = amounts[-1]
    else:
        # Only one amount found; treat as full
        full = amounts[0]
    return discounted, full


def parse_due_dates(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to read discount deadline and overall due date:
    - "within 14 days ... by 28/03/2025"
    - "payment ... within 28 days of the date issued: by 12/04/2025"
    """
    # Any explicit "by DD/MM/YYYY"
    by_dates = re.findall(r"\bby\s+(\d{1,2}/\d{1,2}/\d{4})\b", text, re.IGNORECASE)
    by_dates_norm = [to_uk_date(d) for d in by_dates if to_uk_date(d)]

    discount_deadline = None
    due_date = None

    # simple heuristic: first date near "14 days" is discount; any "28 days" near by-date is due
    # Use windowing around "14" or "28"
    t = text.lower()

    # discount
    m14 = re.search(r"14\s+days", t)
    if m14 and by_dates_norm:
        # take the closest by-date after the phrase
        idx = m14.end()
        closest = None
        closest_dist = 9999
        for m in re.finditer(r"\bby\s+(\d{1,2}/\d{1,2}/\d{4})\b", t):
            dist = abs(m.start() - idx)
            if dist < closest_dist:
                dt_norm = to_uk_date(m.group(1))
                if dt_norm:
                    closest = dt_norm
                    closest_dist = dist
        discount_deadline = closest

    # due: look for "28 days" or generic "payment ... by"
    m28 = re.search(r"28\s+days", t)
    if m28 and by_dates_norm:
        idx = m28.end()
        closest = None
        closest_dist = 9999
        for m in re.finditer(r"\bby\s+(\d{1,2}/\d{1,2}/\d{4})\b", t):
            dist = abs(m.start() - idx)
            if dist < closest_dist:
                dt_norm = to_uk_date(m.group(1))
                if dt_norm:
                    closest = dt_norm
                    closest_dist = dist
        due_date = closest

    # if we still don't have due_date but we have any by-date, take the last one
    if not due_date and by_dates_norm:
        due_date = by_dates_norm[-1]

    return discount_deadline, due_date


def parse_authority(text: str) -> Optional[str]:
    # Look for lines mentioning council/borough/limited/parking services
    candidates = []
    for line in text.splitlines():
        ln = line.strip()
        if re.search(r"(council|borough|parking.*services|limited|ltd)", ln, re.IGNORECASE):
            # Avoid generic phrases like "parking charge amount due"
            if len(ln) > 3 and "charge" not in ln.lower():
                candidates.append(normalise_whitespace(ln))
    # Prefer those with 'council' or 'limited/ltd'
    for kw in ("council", "borough", "limited", "ltd"):
        for c in candidates:
            if kw in c.lower():
                return c
    return candidates[0] if candidates else None


def parse_location(text: str) -> Optional[str]:
    # Prefer explicit "Location:" line
    for line in text.splitlines():
        if "location" in line.lower():
            after = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
            if len(after) > 3:
                return normalise_whitespace(after)

    # Else try to find a UK postcode-ish substring in context lines
    postcode = re.search(r"\b[A-Z]{1,2}\d[0-9A-Z]?\s*\d[A-Z]{2}\b", text, re.IGNORECASE)
    if postcode:
        # return the whole line containing it
        for line in text.splitlines():
            if postcode.group(0) in line:
                return normalise_whitespace(line.strip())

    return None


def parse_pcn_number(text: str) -> Optional[str]:
    # Look for "PCN reference/number/ref: <alnum>"
    m = re.search(r"pcn[^A-Za-z0-9]{0,8}(reference|number|ref)[^A-Za-z0-9]{0,5}([A-Z0-9\-]{6,})", text, re.IGNORECASE)
    if m:
        return m.group(2).upper()
    # Fallback: long uppercase alnum token next to "PCN"
    m2 = re.search(r"\bPCN\b.*?\b([A-Z0-9]{6,})\b", text, re.IGNORECASE | re.DOTALL)
    if m2:
        return m2.group(1).upper()
    return None


def parse_contravention(text: str) -> Tuple[Optional[str], Optional[str]]:
    # Look for "Contravention code" or "Reason"
    code = None
    contr_type = None

    for line in text.splitlines():
        if re.search(r"(contravention|reason|code)", line, re.IGNORECASE):
            m = re.search(r"\b(\d{2}[A-Z]?)\b", line)
            if m:
                code = m.group(1)

    if code and code in CONTRAVENTION_TYPES:
        contr_type = CONTRAVENTION_TYPES[code]
    else:
        # Look for phrase hints
        if re.search(r"no\s+valid\s+parking\s+(ticket|session)", text, re.IGNORECASE):
            contr_type = "No valid parking ticket"
        elif re.search(r"disabled\s+bay", text, re.IGNORECASE):
            contr_type = "Parking in disabled bay"

    return code, contr_type


def pick_best_vrm(text: str) -> Tuple[Optional[str], float]:
    # Find candidates, then prefer those near labels
    cands = vrm_candidates(text)
    if not cands:
        return None, 0.0

    labels = ["vehicle registration", "registration mark", "vrm", "vehicle reg", " licence plate", "license plate"]
    best = None
    best_score = -999
    for c in cands:
        s = 0
        s += 2 if len(c) in (7, 8) else 1  # prefer standard format length
        s += score_near_label(text, labels, c)
        if s > best_score:
            best = c
            best_score = s

    conf = 0.6 + 0.1 * min(3, best_score)  # simple bounded confidence
    return best, min(0.95, conf)


def preprocess_pil(im: Image.Image) -> Image.Image:
    """PIL-only fallback preprocessing."""
    img = im.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.invert(ImageOps.autocontrast(ImageOps.invert(img)))
    # upscale a bit to help Tesseract
    w, h = img.size
    scale = 1.5 if max(w, h) < 2000 else 1.0
    if scale != 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img


def preprocess_cv(im: Image.Image) -> Image.Image:
    """Better preprocessing with OpenCV when available."""
    if not HAS_CV:
        return preprocess_pil(im)

    arr = np.array(im.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # denoise a bit
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    # slight dilation/erosion to connect chars
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # upscale if small
    h, w = th.shape
    scale = 1.5 if max(w, h) < 2000 else 1.0
    if scale != 1.0:
        th = cv2.resize(th, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(th)


def ocr_image(pil_img: Image.Image) -> str:
    return pytesseract.image_to_string(pil_img, config=TESS_CONFIG)


# ------------------------- EXTRACTION PIPELINE -------------------------

def extract_fields(full_text: str) -> dict:
    text = normalise_whitespace(full_text)
    # Keep line structure too
    lines_text = full_text

    # VRM
    vrm, vrm_conf = pick_best_vrm(text)

    # PCN
    pcn = parse_pcn_number(lines_text)

    # Dates (contravention/issue)
    # Try to find a date near keywords first
    contravention_date = None
    for line in lines_text.splitlines():
        if re.search(r"(date\s+of\s+(issue|contravention)|contravention\s+date|offence\s+date)", line, re.IGNORECASE):
            cand = to_uk_date(line)
            if cand:
                contravention_date = cand
                break
    if not contravention_date:
        contravention_date = to_uk_date(text)  # first date anywhere

    # Amounts
    fine_discounted, fine_full = parse_amounts(text)

    # Deadlines
    discount_deadline, due_date = parse_due_dates(text)

    # Authority
    authority = parse_authority(lines_text)

    # Location
    location = parse_location(lines_text)

    # Contravention
    code, contr_type = parse_contravention(lines_text)

    # Confidence (lightweight heuristic)
    conf = 0.0
    conf += 0.2 if vrm else 0.0
    conf += 0.15 if fine_full else 0.0
    conf += 0.1 if (discount_deadline or due_date) else 0.0
    conf += 0.15 if (authority or location) else 0.0
    conf += 0.1 if pcn else 0.0
    conf = min(0.95, max(0.0, conf))

    return {
        "pcn_number": pcn,
        "vrm": vrm,
        "contravention_date": contravention_date,
        "contravention_code": code,
        "contravention_type": contr_type,
        "location": location,
        "authority": authority,
        "fine_amount_discounted": fine_discounted,
        "fine_amount_full": fine_full,
        "discount_deadline": discount_deadline,
        "due_date": due_date,
        "confidence": round(conf, 2),
    }


# ------------------------------ ENDPOINT ------------------------------

@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Render PDF at higher DPI for better OCR
        images: List[Image.Image]
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(content, dpi=300)
        else:
            images = [Image.open(io.BytesIO(content))]

        full_text_pages = []
        for img in images:
            # Preprocess
            prep = preprocess_cv(img)
            # OCR
            txt = ocr_image(prep)
            full_text_pages.append(txt)

        full_text = "\n".join(full_text_pages)

        # Extract fields
        data = extract_fields(full_text)

        return JSONResponse(
            content={
                "success": True,
                "message": "Data extracted successfully",
                "data": data,
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import re
import io
import math
from datetime import datetime
from typing import Optional, Tuple, List

# Optional OpenCV (better preprocessing). Falls back to PIL if not installed.
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    HAS_CV = True
except Exception:
    HAS_CV = False

app = FastAPI(title="OCR Microservice")

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


# ------------------------- CONFIG & HELPERS -------------------------

TESS_CONFIG = r'--oem 3 --psm 6 -l eng'

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

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def normalise_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())


def to_uk_date(s: str) -> Optional[str]:
    """Return DD/MM/YYYY when possible from a variety of formats."""
    s = s.strip()
    # 1) DD/MM/YYYY
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", s)
    if m:
        d, mth, y = m.groups()
        try:
            dt = datetime(int(y), int(mth), int(d))
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    # 2) D/M/YYYY (single-digit day or month)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s)
    if m:
        d, mth, y = m.groups()
        try:
            dt = datetime(int(y), int(mth), int(d))
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    # 3) "12 April 2025" style
    m = re.search(
        r"\b(\d{1,2})[^\w]?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s*(\d{4})\b",
        s, re.IGNORECASE)
    if m:
        d, mon, y = m.groups()
        mon_num = MONTHS.get(mon.lower())
        try:
            dt = datetime(int(y), mon_num, int(d))
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    return None


def vrm_candidates(text: str) -> List[str]:
    """
    Try to find UK-style VRMs. Prioritize those near VRM labels later.
    This regex is intentionally permissive; we'll post-filter.
    """
    patterns = [
        r"\b[A-Z]{2}\s?\d{2}\s?[A-Z]{3}\b",      # current style AB12 CDE
        r"\b[A-Z]{1,3}\s?\d{1,4}\b",            # older styles
        r"\b[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}\b"
    ]
    cands = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            cands.add(m.group(0).replace(" ", "").upper())
    # Filter out likely noise
    filtered = []
    for c in cands:
        if 4 <= len(c) <= 8 and not c.startswith("PCN"):
            filtered.append(c)
    return filtered


def score_near_label(text: str, label_keywords: List[str], value: str) -> int:
    """
    Simple proximity score: +1 if value appears within N chars of any label keyword.
    """
    t = text.lower()
    pos_val = t.find(value.lower())
    best = 0
    if pos_val == -1:
        return best
    for kw in label_keywords:
        pos_kw = t.find(kw)
        if pos_kw != -1:
            dist = abs(pos_kw - pos_val)
            if dist < 120:  # within ~120 chars = likely same block/line cluster
                best = max(best, 1)
    return best


def parse_amounts(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (discounted, full) based on £ amounts, smallest as discounted if text mentions discount."""
    amounts = [float(m.replace(",", "")) for m in re.findall(r"£\s*([\d,]+(?:\.\d{2})?)", text)]
    if not amounts:
        return None, None
    amounts = sorted(set(amounts))
    # heuristics
    discounted, full = None, None
    has_discount_words = re.search(r"discount|within\s+14|reduced", text, re.IGNORECASE)

    if has_discount_words and len(amounts) >= 2:
        discounted = amounts[0]
        full = amounts[-1]
    elif len(amounts) >= 2:
        # pick smallest as discounted, largest as full
        discounted = amounts[0]
        full = amounts[-1]
    else:
        # Only one amount found; treat as full
        full = amounts[0]
    return discounted, full


def parse_due_dates(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to read discount deadline and overall due date:
    - "within 14 days ... by 28/03/2025"
    - "payment ... within 28 days of the date issued: by 12/04/2025"
    """
    # Any explicit "by DD/MM/YYYY"
    by_dates = re.findall(r"\bby\s+(\d{1,2}/\d{1,2}/\d{4})\b", text, re.IGNORECASE)
    by_dates_norm = [to_uk_date(d) for d in by_dates if to_uk_date(d)]

    discount_deadline = None
    due_date = None

    # simple heuristic: first date near "14 days" is discount; any "28 days" near by-date is due
    # Use windowing around "14" or "28"
    t = text.lower()

    # discount
    m14 = re.search(r"14\s+days", t)
    if m14 and by_dates_norm:
        # take the closest by-date after the phrase
        idx = m14.end()
        closest = None
        closest_dist = 9999
        for m in re.finditer(r"\bby\s+(\d{1,2}/\d{1,2}/\d{4})\b", t):
            dist = abs(m.start() - idx)
            if dist < closest_dist:
                dt_norm = to_uk_date(m.group(1))
                if dt_norm:
                    closest = dt_norm
                    closest_dist = dist
        discount_deadline = closest

    # due: look for "28 days" or generic "payment ... by"
    m28 = re.search(r"28\s+days", t)
    if m28 and by_dates_norm:
        idx = m28.end()
        closest = None
        closest_dist = 9999
        for m in re.finditer(r"\bby\s+(\d{1,2}/\d{1,2}/\d{4})\b", t):
            dist = abs(m.start() - idx)
            if dist < closest_dist:
                dt_norm = to_uk_date(m.group(1))
                if dt_norm:
                    closest = dt_norm
                    closest_dist = dist
        due_date = closest

    # if we still don't have due_date but we have any by-date, take the last one
    if not due_date and by_dates_norm:
        due_date = by_dates_norm[-1]

    return discount_deadline, due_date


def parse_authority(text: str) -> Optional[str]:
    # Look for lines mentioning council/borough/limited/parking services
    candidates = []
    for line in text.splitlines():
        ln = line.strip()
        if re.search(r"(council|borough|parking.*services|limited|ltd)", ln, re.IGNORECASE):
            # Avoid generic phrases like "parking charge amount due"
            if len(ln) > 3 and "charge" not in ln.lower():
                candidates.append(normalise_whitespace(ln))
    # Prefer those with 'council' or 'limited/ltd'
    for kw in ("council", "borough", "limited", "ltd"):
        for c in candidates:
            if kw in c.lower():
                return c
    return candidates[0] if candidates else None


def parse_location(text: str) -> Optional[str]:
    # Prefer explicit "Location:" line
    for line in text.splitlines():
        if "location" in line.lower():
            after = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
            if len(after) > 3:
                return normalise_whitespace(after)

    # Else try to find a UK postcode-ish substring in context lines
    postcode = re.search(r"\b[A-Z]{1,2}\d[0-9A-Z]?\s*\d[A-Z]{2}\b", text, re.IGNORECASE)
    if postcode:
        # return the whole line containing it
        for line in text.splitlines():
            if postcode.group(0) in line:
                return normalise_whitespace(line.strip())

    return None


def parse_pcn_number(text: str) -> Optional[str]:
    # Look for "PCN reference/number/ref: <alnum>"
    m = re.search(r"pcn[^A-Za-z0-9]{0,8}(reference|number|ref)[^A-Za-z0-9]{0,5}([A-Z0-9\-]{6,})", text, re.IGNORECASE)
    if m:
        return m.group(2).upper()
    # Fallback: long uppercase alnum token next to "PCN"
    m2 = re.search(r"\bPCN\b.*?\b([A-Z0-9]{6,})\b", text, re.IGNORECASE | re.DOTALL)
    if m2:
        return m2.group(1).upper()
    return None


def parse_contravention(text: str) -> Tuple[Optional[str], Optional[str]]:
    # Look for "Contravention code" or "Reason"
    code = None
    contr_type = None

    for line in text.splitlines():
        if re.search(r"(contravention|reason|code)", line, re.IGNORECASE):
            m = re.search(r"\b(\d{2}[A-Z]?)\b", line)
            if m:
                code = m.group(1)

    if code and code in CONTRAVENTION_TYPES:
        contr_type = CONTRAVENTION_TYPES[code]
    else:
        # Look for phrase hints
        if re.search(r"no\s+valid\s+parking\s+(ticket|session)", text, re.IGNORECASE):
            contr_type = "No valid parking ticket"
        elif re.search(r"disabled\s+bay", text, re.IGNORECASE):
            contr_type = "Parking in disabled bay"

    return code, contr_type


def pick_best_vrm(text: str) -> Tuple[Optional[str], float]:
    # Find candidates, then prefer those near labels
    cands = vrm_candidates(text)
    if not cands:
        return None, 0.0

    labels = ["vehicle registration", "registration mark", "vrm", "vehicle reg", " licence plate", "license plate"]
    best = None
    best_score = -999
    for c in cands:
        s = 0
        s += 2 if len(c) in (7, 8) else 1  # prefer standard format length
        s += score_near_label(text, labels, c)
        if s > best_score:
            best = c
            best_score = s

    conf = 0.6 + 0.1 * min(3, best_score)  # simple bounded confidence
    return best, min(0.95, conf)


def preprocess_pil(im: Image.Image) -> Image.Image:
    """PIL-only fallback preprocessing."""
    img = im.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.invert(ImageOps.autocontrast(ImageOps.invert(img)))
    # upscale a bit to help Tesseract
    w, h = img.size
    scale = 1.5 if max(w, h) < 2000 else 1.0
    if scale != 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img


def preprocess_cv(im: Image.Image) -> Image.Image:
    """Better preprocessing with OpenCV when available."""
    if not HAS_CV:
        return preprocess_pil(im)

    arr = np.array(im.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # denoise a bit
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    # slight dilation/erosion to connect chars
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # upscale if small
    h, w = th.shape
    scale = 1.5 if max(w, h) < 2000 else 1.0
    if scale != 1.0:
        th = cv2.resize(th, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(th)


def ocr_image(pil_img: Image.Image) -> str:
    return pytesseract.image_to_string(pil_img, config=TESS_CONFIG)


# ------------------------- EXTRACTION PIPELINE -------------------------

def extract_fields(full_text: str) -> dict:
    text = normalise_whitespace(full_text)
    # Keep line structure too
    lines_text = full_text

    # VRM
    vrm, vrm_conf = pick_best_vrm(text)

    # PCN
    pcn = parse_pcn_number(lines_text)

    # Dates (contravention/issue)
    # Try to find a date near keywords first
    contravention_date = None
    for line in lines_text.splitlines():
        if re.search(r"(date\s+of\s+(issue|contravention)|contravention\s+date|offence\s+date)", line, re.IGNORECASE):
            cand = to_uk_date(line)
            if cand:
                contravention_date = cand
                break
    if not contravention_date:
        contravention_date = to_uk_date(text)  # first date anywhere

    # Amounts
    fine_discounted, fine_full = parse_amounts(text)

    # Deadlines
    discount_deadline, due_date = parse_due_dates(text)

    # Authority
    authority = parse_authority(lines_text)

    # Location
    location = parse_location(lines_text)

    # Contravention
    code, contr_type = parse_contravention(lines_text)

    # Confidence (lightweight heuristic)
    conf = 0.0
    conf += 0.2 if vrm else 0.0
    conf += 0.15 if fine_full else 0.0
    conf += 0.1 if (discount_deadline or due_date) else 0.0
    conf += 0.15 if (authority or location) else 0.0
    conf += 0.1 if pcn else 0.0
    conf = min(0.95, max(0.0, conf))

    return {
        "pcn_number": pcn,
        "vrm": vrm,
        "contravention_date": contravention_date,
        "contravention_code": code,
        "contravention_type": contr_type,
        "location": location,
        "authority": authority,
        "fine_amount_discounted": fine_discounted,
        "fine_amount_full": fine_full,
        "discount_deadline": discount_deadline,
        "due_date": due_date,
        "confidence": round(conf, 2),
    }


# ------------------------------ ENDPOINT ------------------------------

@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Render PDF at higher DPI for better OCR
        images: List[Image.Image]
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(content, dpi=300)
        else:
            images = [Image.open(io.BytesIO(content))]

        full_text_pages = []
        for img in images:
            # Preprocess
            prep = preprocess_cv(img)
            # OCR
            txt = ocr_image(prep)
            full_text_pages.append(txt)

        full_text = "\n".join(full_text_pages)

        # Extract fields
        data = extract_fields(full_text)

        return JSONResponse(
            content={
                "success": True,
                "message": "Data extracted successfully",
                "data": data,
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
