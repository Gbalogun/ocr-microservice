# OCR Microservice for PCN Automation

This microservice extracts structured data from scanned Penalty Charge Notices (PCNs) using OCR. It supports both image and PDF formats and is designed to integrate with a PCN automation SaaS platform.

---

## 🧠 Features

- Accepts `.jpg`, `.png`, and `.pdf` uploads
- Uses Tesseract OCR to extract text
- Parses and returns key PCN fields:
  - Vehicle Registration Mark (VRM)
  - Contravention Date
  - Contravention Code
  - Location (basic placeholder)
  - Issuing Authority (basic placeholder)

---

## 🚀 API Endpoint

### `POST /ocr`

**Form field:** `file`  
**Content-Type:** `multipart/form-data`

#### Example Response:
```json
{
  "vrm": "AB12CDE",
  "contravention_date": "15/09/2025",
  "location": null,
  "authority": null,
  "contravention_code": "34J"
}
