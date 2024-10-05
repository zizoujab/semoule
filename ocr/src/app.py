from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Step 1: Load the image file from the request
        image_file = request.files['image']

        # Convert the image file to bytes
        image_bytes = image_file.read()

        # Step 2: Load the image using OpenCV and convert it to grayscale
        np_image = np.frombuffer(image_bytes, np.uint8)  # Convert bytes to NumPy array
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)  # Decode image from bytes

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply thresholding to get a binary image
        _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

        # Optional: You can apply additional preprocessing if needed
        # Example: Gaussian blur
        # blurred_image = cv2.GaussianBlur(thresh_image, (5, 5), 0)

        # Convert the image to PIL format for pytesseract
        pil_image = Image.fromarray(thresh_image)

        # Step 4: Perform OCR using pytesseract
        extracted_text = pytesseract.image_to_string(pil_image)

        return jsonify({'extracted_text': extracted_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)