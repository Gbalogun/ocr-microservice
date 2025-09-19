# OCR Microservice for PCN Automation

This service extracts structured data from scanned PCNs (Penalty Charge Notices) using OCR.
It supports both **images** and **PDFs**.

## Features
- Extracts VRM, contravention date, and code from PCNs
- FastAPI-based API
- Works with Tesseract OCR + Poppler

## Endpoints
- `GET /` → Health check
- `POST /ocr` → Upload PCN image/PDF → Returns extracted JSON
- `GET /docs` → Swagger UI

## Deploy on Render
1. Push this repo to GitHub
2. On Render, create a **New Web Service**
3. Select **Docker** environment
4. Deploy 🚀
