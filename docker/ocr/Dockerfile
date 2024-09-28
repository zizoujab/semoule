# Use Python slim as a base
FROM python:3.9-slim

# Install Tesseract OCR and other dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    python3-pip \
    && apt-get clean

# Install Python dependencies including Flask
RUN pip install pytesseract Pillow Flask
# Set the working directory (this will be where the volume is mounted)
WORKDIR /app/src

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run the Flask app (app.py will be mounted from your local machine)
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
