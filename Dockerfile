# 1. Start from a base Python image
FROM python:3.10-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Install system dependencies
# This is for Tesseract (OCR) and Poppler (PDF-to-Image)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 4. Set a working directory in the container
WORKDIR /app

# 5. Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your app's code
COPY . .

# 7. Expose the port the app will run on (set by Cloud Run)
ENV PORT 8080

# 8. Define the command to run the app
# We use Gunicorn, a production-ready web server
# It will run the 'app' object from the 'chat_with_pdf_v1' file
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "chat_with_pdf_v1:app"]
