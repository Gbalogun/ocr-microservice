import os
import base64
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ================================
# ✅ INITIAL SETUP
# ================================

app = FastAPI()

# Allow frontend (Softgen / local) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================
# ✅ HEALTH CHECK ROUTE
# ================================

@app.get("/")
def health_check():
    """Basic check to confirm the service is running."""
    return {"status": "ok", "message": "OCR service active"}

# ================================
# ✅ OCR EXTRACTION ENDPOINT
# ================================

@app.post("/ocr")
async def extract_pcn_data(file: UploadFile = File(...)):
    """
    Accepts a PDF or image file, sends it to OpenAI Vision (gpt-4o-mini),
    and extracts structured PCN data.
    """
    try:
        # Read and encode uploaded file
        file_bytes = await file.read()
        encoded = base64.b64encode(file_bytes).decode("utf-8")

        # Define clear extraction prompt
        prompt = (
            "Extract the following information from this penalty charge notice (PCN):\n"
            " - VRM (vehicle registration number)\n"
            " - Contravention date (DD/MM/YYYY)\n"
            " - Location\n"
            " - Issuing authority\n"
            " - Fine amount (include £ sign if present)\n"
            " - PCN number / reference\n"
            " - Reason for fine\n\n"
            "Return only valid JSON with these exact keys:\n"
            "vrm, contravention_date, location, authority, fine_amount, pcn_number, reason."
        )

        # Call OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts PCN data from documents."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{encoded}"}}
                ]},
            ],
            temperature=0.2,
        )

        # Parse the response
        result_text = response.choices[0].message.content.strip()

        # Return structured response
        return {"success": True, "message": "Data extracted successfully", "data": result_text}

    except Exception as e:
        # Log detailed error to console for debugging (visible in Render logs)
        print("❌ OCR extraction error:", e)
        traceback.print_exc()

        # Return a safe JSON error message
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
