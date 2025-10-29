# main.py
import os
import io
import re
import json
import base64
import tempfile
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_bytes
import pytesseract

from openai import OpenAI, OpenAIError

# ────────────────────────────────────────────────────────────────────────────────
# App & CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="OCR Microservice (OpenAI Vision + PDF support)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

RESPONSE_TEMPLATE = {
    "pcn_number": None,
    "vrm": None,
    "contravention_date": None,   # DD/MM/YYYY
    "contravention_code": None,
    "contravention_type": None,
    "location": None,
    "authority": None,
    "fine_amount_discounted": None,  # number
    "fine_amount_full": None,        # number
    "discount_deadline": None,       # DD/MM/YYYY
    "due_date": None,                # DD/MM/YYYY
    "confidence": 0.0,
}

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

def image_to_base64(img: Image.Image) -> str:
    """Encode PIL image to base64 (JPEG)."""
    buff = io.BytesIO()
    # ensure RGB
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    else:
        img = img.convert("RGB")
    img.save(buff, format="JPEG", quality=85)
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def convert_pdf_to_images(file_bytes: bytes, max_pages: int = 3) -> List[Image.Image]:
    """
    Convert PDF bytes into a list of PIL images using poppler (pdftoppm).
    Requires `poppler-utils` on the server.
    """
    try:
        pages = convert_from_bytes(file_bytes, fmt="jpeg")
        if not pages:
            raise RuntimeError("No pages produced from PDF conversion.")
        return pages[:max_pages]  # use first few pages
    except Exception as e:
        raise RuntimeError(
            f"PDF conversion failed. Ensure poppler-utils is installed. Details: {e}"
        )

def simple_regex_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback extraction using regex on raw OCR text (pytesseract).
    This is a last resort if OpenAI Vision call fails.
    """
    data = RESPONSE_TEMPLATE.copy()

    # VRM (UK style: AB12CDE)
    m = re.search(r"\b([A-Z]{2}\d{2}[A-Z]{3})\b", text)
    if m:
        data["vrm"] = m.group(1)

    # Dates — UK dd/mm/yyyy
    dates = re.findall(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    if dates:
        data["contravention_date"] = dates[0]
        if len(dates) > 1:
            data["due_date"] = dates[-1]

    # Contravention code
    m = re.search(r"\b(\d{2}[A-Z]?)\b", text)
    if m:
        code = m.group(1)
        data["contravention_code"] = code
        data["contravention_type"] = CONTRAVENTION_TYPES.get(code)

    # PCN reference / number
    m = re.search(r"(PCN\s*(?:Ref|Reference|Number|No\.?)\s*[:#]?\s*([A-Z0-9\-]+))", text, re.I)
    if m:
        data["pcn_number"] = m.group(2)

    # Authority (look for council/limited)
    m = re.search(r"(council|borough|limited|ltd)", text, re.I)
    if m:
        # crude: take a line with that word
        for line in text.splitlines():
            if re.search(r"(council|borough|limited|ltd)", line, re.I):
                data["authority"] = line.strip()
                break

    # Location – crude fallback: line with a UK postcode-like token
    m = re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", text)
    if m:
        # Use the line that includes that postcode token
        for line in text.splitlines():
            if m.group(0) in line:
                data["location"] = line.strip()
                break

    # Fine amounts
    amounts = [float(a.replace(",", "")) for a in re.findall(r"£\s?(\d+(?:\.\d{2})?)", text)]
    if amounts:
        data["fine_amount_full"] = max(amounts)
        data["fine_amount_discounted"] = min(amounts) if len(amounts) > 1 else None

    data["confidence"] = 0.35  # we don’t trust this much
    return data

def merge_defaults(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure all expected keys exist."""
    merged = RESPONSE_TEMPLATE.copy()
    if d:
        for k, v in d.items():
            if k in merged:
                merged[k] = v
    return merged

def build_openai_prompt() -> str:
    """Prompt that forces strict JSON with desired fields."""
    return (
        "You are a document extraction engine. Extract fields from a UK private parking PCN. "
        "Return STRICT JSON ONLY (no prose) with this exact shape and keys (string values or null):\n\n"
        "{\n"
        '  "pcn_number": string|null,\n'
        '  "vrm": string|null,\n'
        '  "contravention_date": string|null,        // format DD/MM/YYYY\n'
        '  "contravention_code": string|null,\n'
        '  "contravention_type": string|null,        // e.g., "No valid parking ticket"\n'
        '  "location": string|null,\n'
        '  "authority": string|null,                 // issuer (council, borough or LTD)\n'
        '  "fine_amount_discounted": number|null,    // lower £ value\n'
        '  "fine_amount_full": number|null,          // higher £ value\n'
        '  "discount_deadline": string|null,         // DD/MM/YYYY if present\n'
        '  "due_date": string|null,                  // DD/MM/YYYY if present\n'
        '  "confidence": number                      // 0-1 estimation\n'
        "}\n\n"
        "Rules:\n"
        "- Always use UK date format DD/MM/YYYY when you infer a date.\n"
        "- If multiple £ amounts exist, the smaller is discounted, the larger is full amount.\n"
        "- If a contravention code maps to a known type (01,02,03,04,05,06,12,16,23), include the friendly name.\n"
        "- If not present, use null.\n"
        "- Return ONLY JSON. No extra text."
    )

def call_openai_vision(images: List[Image.Image]) -> Dict[str, Any]:
    """
    Send up to 3 page images to OpenAI Vision for structured extraction.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    # Build OpenAI content array: text prompt + one image per page
    content_parts = [{"type": "text", "text": build_openai_prompt()}]
    for img in images[:3]:
        b64 = image_to_base64(img)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    # Use chat.completions with gpt-4o-mini (vision + JSON-friendly)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content_parts}],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Ensure it's valid JSON (strip code fences if model added any)
        raw = raw.strip("` \n")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        data = json.loads(raw)
        return merge_defaults(data)
    except (OpenAIError, json.JSONDecodeError) as e:
        raise RuntimeError(f"OpenAI Vision extraction failed: {e}")

def ocr_text_with_tesseract(images: List[Image.Image]) -> str:
    text = []
    for img in images:
        try:
            text.append(pytesseract.image_to_string(img))
        except Exception:
            continue
    return "\n".join(text)

# ────────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"status": "ok", "message": "OCR microservice (OpenAI Vision) is running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """
    Accepts a PDF/image, extracts PCN fields using OpenAI Vision.
    Falls back to Tesseract+regex if the OpenAI call fails.
    """
    try:
        content = await file.read()

        # Convert to list[Image.Image]
        images: List[Image.Image] = []
        fname = (file.filename or "").lower()

        if fname.endswith(".pdf") or content[:4] == b"%PDF":
            # PDF → images (requires poppler-utils)
            images = convert_pdf_to_images(content, max_pages=3)
        else:
            # Try image
            try:
                im = Image.open(io.BytesIO(content))
                images = [im]
            except UnidentifiedImageError:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Unsupported file type (not PDF or image)."}
                )

        # First try OpenAI Vision
        try:
            extracted = call_openai_vision(images)
            return JSONResponse(content={"success": True, "data": extracted})
        except Exception as openai_err:
            # Fallback: Tesseract + regex
            text = ocr_text_with_tesseract(images)
            fallback = simple_regex_fallback(text)
            return JSONResponse(
                content={
                    "success": True,
                    "data": fallback,
                    "warning": f"OpenAI extraction failed; used fallback. Details: {str(openai_err)}"
                }
            )

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
