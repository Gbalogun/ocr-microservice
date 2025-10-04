from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import re
import io
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# --- Optional imports (auto-disabled if not available) ---
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

# ---------------------------------------------------------

app = FastAPI(title="OCR Microservice (UK PCN)")

# CORS
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

# ----------------------------
# Config / Dictionaries
# ----------------------------

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

AUTHORITY_HINTS = [
    "council", "borough", "parking services", "parking management", "parking control",
    "parking charge", "parking eye", "euro car parks", "civil enforcement", "pcn admin",
    "ltd", "limited", "uk ltd", "group", "services ltd", "parking", "maven"
]

# ----------------------------
# Helpers
# ----------------------------

def norm_text(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())

def split_lines(text: str) -> List[str]:
    return [norm_text(l) for l in text.splitlines() if norm_text(l)]

def within(idx: int, n: int) -> List[int]:
    # return indices for a small window centered at idx
    cand = {idx}
    if idx - 1 >= 0: cand.add(idx - 1)
    if idx + 1 < n: cand.add(idx + 1)
    if idx - 2 >= 0: cand.add(idx - 2)
    if idx + 2 < n: cand.add(idx + 2)
    return sorted(cand)

def parse_date_uk(s: str) -> Optional[str]:
    # Accept DD/MM/YYYY or DD-MM-YYYY; return normalized DD/MM/YYYY
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            d = datetime.strptime(s, fmt)
            return d.strftime("%d/%m/%Y")
        except Exception:
            pass
    return None

def is_vrm(s: str) -> bool:
    # Robust UK VRM (current style)
    return re.fullmatch(r"[A-Z]{2}\d{2}[A-Z]{3}", s) is not None

def clean_vrm(s: str) -> str:
    # Fix common OCR confusions for VRM-like strings
    subs = {'0':'O', '1':'I', '5':'S', '8':'B'}
    s = s.upper().strip()
    if len(s) == 7:
        s = "".join(subs.get(ch, ch) for ch in s)
    return s

def best_amounts_from_lines(lines: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Returns (discounted, full, discount_deadline)
    Uses proximity/wording to infer roles.
    """
    discounted = None
    full = None
    discount_deadline = None

    amount_pat = re.compile(r"£\s?(\d+(?:\.\d{2})?)")
    date_pat = re.compile(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", re.IGNORECASE)

    all_amounts = []  # (value, idx, line)
    for i, line in enumerate(lines):
        for m in amount_pat.findall(line):
            try:
                v = float(m)
                all_amounts.append((v, i, line))
            except Exception:
                pass

    if not all_amounts:
        return (None, None, None)

    # classify per line semantics
    for v, i, line in all_amounts:
        ll = line.lower()
        # discount hint
        if "discount" in ll or "within 14" in ll or "14 days" in ll:
            discounted = v if (discounted is None or v < discounted) else discounted
            # check a date in or around line for discount deadline
            window = within(i, len(lines))
            for j in window:
                for dm in date_pat.findall(lines[j]):
                    d = parse_date_uk(dm)
                    if d:
                        discount_deadline = d
                        break
                if discount_deadline:
                    break
        # full/amount due hints
        if any(k in ll for k in ["amount due", "parking charge amount", "full amount", "amount:"]):
            full = v if (full is None or v > full) else full

    # If still ambiguous, use min as discount, max as full
    if all_amounts and (discounted is None or full is None):
        values = [v for v, _, _ in all_amounts]
        mn, mx = min(values), max(values)
        if discounted is None:
            discounted = mn
        if full is None:
            full = mx

    return (discounted, full, discount_deadline)

def find_due_dates(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (due_date, contravention_date)
    Uses line semantics: 'payment within/by' -> due,
    'issue/offence/contravention' -> contravention.
    """
    date_pat = re.compile(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", re.IGNORECASE)
    due_date = None
    contravention_date = None

    for i, line in enumerate(lines):
        ll = line.lower()
        # date tokens on this line or neighbors
        window = within(i, len(lines))
        dates = []
        for j in window:
            dates += date_pat.findall(lines[j])

        if not dates:
            continue

        # classify
        if any(k in ll for k in ["payment", "to be made", "within 28 days", "by "]):
            for d in dates:
                nd = parse_date_uk(d)
                if nd:
                    due_date = nd
                    break

        if any(k in ll for k in ["contravention", "issue", "offence", "date of issue"]):
            for d in dates:
                nd = parse_date_uk(d)
                if nd:
                    contravention_date = nd
                    break

    return (due_date, contravention_date)

def fuzzy_contains_any(s: str, keys: List[str]) -> bool:
    ls = s.lower()
    return any(k in ls for k in keys)

def extract_location(lines: List[str]) -> Optional[str]:
    # Prefer explicit "Location:"; else a line with a UK postcode
    for line in lines:
        if "location" in line.lower():
            return norm_text(line.split(":", 1)[-1])

    # UK postcode heuristic
    for line in lines:
        if re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", line):
            return norm_text(line)
    return None

def extract_authority(lines: List[str]) -> Optional[str]:
    # Pick the most "organization shaped" line using hints.
    candidates = []
    for line in lines:
        if fuzzy_contains_any(line, AUTHORITY_HINTS):
            candidates.append(line)
    if not candidates:
        return None
    # Prefer longer
    cand = max(candidates, key=lambda x: len(x))
    return norm_text(cand)

def extract_pcn_number(lines: List[str]) -> Optional[str]:
    # Look for pcn/reference labels and capture strong alnum token
    lbl = re.compile(r"(pcn|reference|ref(?:erence)?(?:\s*no\.?)?|ticket|notice)\b", re.IGNORECASE)
    tok = re.compile(r"\b[A-Z0-9]{5,}\b")
    for i, line in enumerate(lines):
        if lbl.search(line):
            # same or next line
            window = within(i, len(lines))
            for j in window:
                for m in tok.findall(lines[j].upper()):
                    return m
    # fallback: first long alnum on page (very loose, avoided when possible)
    for line in lines[:10]:
        m = tok.search(line.upper())
        if m:
            return m.group(0)
    return None

def extract_vrm(lines: List[str]) -> Optional[str]:
    # Prefer near labels; otherwise first good VRM.
    lbl = re.compile(r"(vrm|vehicle|reg(?:istration)?|number\s*plate)", re.IGNORECASE)
    vrm_pat = re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{3}\b")
    for i, line in enumerate(lines):
        if lbl.search(line):
            window = within(i, len(lines))
            for j in window:
                for m in vrm_pat.findall(lines[j].upper()):
                    v = clean_vrm(m)
                    if is_vrm(v):
                        return v
    # fallback
    for line in lines:
        for m in vrm_pat.findall(line.upper()):
            v = clean_vrm(m)
            if is_vrm(v):
                return v
    return None

def extract_contravention(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (code, type).
    If code found, map to type; else try to infer type from text near 'contravention/reason/type'
    """
    code_pat = re.compile(r"\b(\d{2}[A-Z]?)\b")
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in ["contravention", "reason", "type"]):
            # find code
            for m in code_pat.findall(line):
                code = m
                return code, CONTRAVENTION_TYPES.get(code, None)
            # No code: try to infer from next/this line
            window_text = " ".join(lines[j] for j in within(i, len(lines)))
            for code, title in CONTRAVENTION_TYPES.items():
                # loose contains
                if all(w.lower() in window_text.lower() for w in title.split()[:2]):  # quick fuzzy
                    return None, title
    return (None, None)

# ----------------------------
# OCR / PDF text extraction
# ----------------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Preferred: use PyMuPDF to extract selectable text.
    Fallback to empty string if not available or failed.
    """
    if not HAS_PYMUPDF:
        return ""
    try:
        text_parts = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)
    except Exception:
        return ""

def enhance_for_ocr(img: Image.Image) -> Image.Image:
    # Grayscale -> autocontrast -> mild sharpen -> binary-ish
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    if HAS_NUMPY:
        arr = np.array(g)
        # Simple threshold
        thr = (arr > 180) * 255
        g = Image.fromarray(thr.astype("uint8"), mode="L")
    return g

def ocr_pages_from_pdf_bytes(pdf_bytes: bytes, dpi: int = 300) -> str:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    text = []
    for p in pages:
        p2 = enhance_for_ocr(p)
        txt = pytesseract.image_to_string(
            p2,
            config="--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1"
        )
        text.append(txt)
    return "\n".join(text)

def ocr_from_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    img = enhance_for_ocr(img)
    return pytesseract.image_to_string(
        img,
        config="--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1"
    )

# ----------------------------
# Main extraction orchestrator
# ----------------------------

def extract_fields_from_text(text: str) -> Tuple[Dict[str, Optional[str]], Dict[str, float], Dict[str, str]]:
    """
    Parse rich text → structured fields.
    Returns (data, confidence, provenance)
    """
    lines = split_lines(text)

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
    conf = {k: 0.0 for k in data.keys()}
    prov = {k: ""   for k in data.keys()}

    # Amounts (discount/full + discount deadline)
    d_amt, f_amt, disc_dead = best_amounts_from_lines(lines)
    if d_amt is not None:
        data["fine_amount_discounted"] = d_amt
        conf["fine_amount_discounted"] = 0.8
        prov["fine_amount_discounted"] = "amount/proximity"
    if f_amt is not None:
        data["fine_amount_full"] = f_amt
        conf["fine_amount_full"] = 0.8
        prov["fine_amount_full"] = "amount/proximity"
    if disc_dead:
        data["discount_deadline"] = disc_dead
        conf["discount_deadline"] = 0.7
        prov["discount_deadline"] = "date/proximity"

    # Due + contravention dates
    due, contr = find_due_dates(lines)
    if due:
        data["due_date"] = due
        conf["due_date"] = 0.8
        prov["due_date"] = "date/semantic"
    if contr:
        data["contravention_date"] = contr
        conf["contravention_date"] = 0.8
        prov["contravention_date"] = "date/semantic"

    # VRM
    vrm = extract_vrm(lines)
    if vrm:
        data["vrm"] = vrm
        conf["vrm"] = 0.9
        prov["vrm"] = "vrm/pattern+label"

    # PCN number
    pcn = extract_pcn_number(lines)
    if pcn:
        data["pcn_number"] = pcn
        conf["pcn_number"] = 0.8
        prov["pcn_number"] = "pcn/label+token"

    # Location
    loc = extract_location(lines)
    if loc:
        data["location"] = loc
        conf["location"] = 0.7
        prov["location"] = "location/label+postcode"

    # Authority
    auth = extract_authority(lines)
    if auth:
        data["authority"] = auth
        conf["authority"] = 0.7
        prov["authority"] = "authority/hints"

    # Contravention code/type
    code, ctype = extract_contravention(lines)
    if code:
        data["contravention_code"] = code
        conf["contravention_code"] = 0.7
        prov["contravention_code"] = "contravention/code"
    if ctype:
        data["contravention_type"] = ctype
        conf["contravention_type"] = 0.6 if not code else 0.8
        prov["contravention_type"] = "contravention/mapping" if code else "contravention/fuzzy"

    return data, conf, prov

# ----------------------------
# Endpoint
# ----------------------------

@app.post("/ocr")
async def ocr_extract(
    file: UploadFile = File(...),
    debug: int = Query(0, description="Set to 1 to include raw_text, confidence and provenance")
):
    try:
        content = await file.read()
        filename = (file.filename or "").lower()

        # 1) Prefer PDF selectable text
        raw_text = ""
        used = "none"
        if filename.endswith(".pdf"):
            if HAS_PYMUPDF:
                raw_text = extract_text_from_pdf_bytes(content)
                if raw_text.strip():
                    used = "pdf_text"
                else:
                    raw_text = ocr_pages_from_pdf_bytes(content, dpi=300)
                    used = "pdf_ocr"
            else:
                raw_text = ocr_pages_from_pdf_bytes(content, dpi=300)
                used = "pdf_ocr"
        else:
            # images
            raw_text = ocr_from_image_bytes(content)
            used = "image_ocr"

        # 2) Parse
        data, conf, prov = extract_fields_from_text(raw_text)

        # Ensure numeric amounts are JSON serializable (float or null)
        for k in ["fine_amount_discounted", "fine_amount_full"]:
            v = data.get(k)
            if v is not None:
                try:
                    data[k] = float(v)
                except Exception:
                    data[k] = None

        # Standard response
        payload = data
        if debug == 1:
            payload = {
                **data,
                "_debug": {
                    "engine": used,
                    "confidence": conf,
                    "provenance": prov,
                    "raw_text": raw_text[:20000],  # cap for safety
                }
            }
        return JSONResponse(content=payload)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
