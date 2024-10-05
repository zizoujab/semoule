# Use Python slim as a base
FROM python:3.9-slim

# Install Tesseract OCR and other dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    python3-pip \
    && apt-get clean

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the working directory
WORKDIR /app/src

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run Flask
CMD ["python", "app.py"]