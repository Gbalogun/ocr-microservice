import base64
from openai import OpenAI
import os
from fastapi.responses import JSONResponse

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_with_openai(file_content: bytes, filename: str):
    """
    Sends the uploaded PCN (PDF or image) to GPT-4o for structured field extraction.
    """

    # Convert file to base64 for multimodal input
    b64 = base64.b64encode(file_content).decode("utf-8")

    prompt = """
    You are an intelligent document parser for UK Penalty Charge Notices (PCNs).
    Carefully read the attached document (PDF or image).
    Extract the following fields and output valid JSON only, with these exact keys:

    {
      "pcn_number": "...",
      "vrm": "...",
      "contravention_date": "...",
      "fine_amount_discounted": "...",
      "fine_amount_full": "...",
      "due_date": "...",
      "authority": "...",
      "location": "...",
      "reason": "..."
    }

    - Dates should be in DD/MM/YYYY format.
    - Fine amounts must include the £ symbol.
    - Authority may include Limited, Borough, or Council names.
    - If a value is missing or unclear, use null.
    """

    try:
        result = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_data": b64},
                    ],
                }
            ],
            temperature=0,
        )

        raw_text = result.output_text.strip()

        # Attempt to return structured JSON output
        return JSONResponse(content={"extracted": raw_text, "engine": "openai"})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "engine": "openai"},
        )
