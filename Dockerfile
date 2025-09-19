# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies: Tesseract OCR + Poppler (for pdf2image)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port (Render will override with $PORT)
EXPOSE 8080

# Start the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
