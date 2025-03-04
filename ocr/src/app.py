from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
import logging

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

#         # Step 4: Detect the orientation of the image using Tesseract's OSD
#         osd = pytesseract.image_to_osd(pil_image)
#         rotation_angle = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
#
#         # Step 5: If the image is rotated, rotate it back to the correct orientation
#         if rotation_angle != 0:
#             # Rotate the image using OpenCV to correct the orientation
#             if rotation_angle == 90:
#                 rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#             elif rotation_angle == 180:
#                 rotated_image = cv2.rotate(image, cv2.ROTATE_180)
#             elif rotation_angle == 270:
#                 rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#             else:
#                 rotated_image = image  # If no recognizable rotation, keep original image
#         else:
#             rotated_image = image  # No rotation needed
        rotated_image = image
        # Step 6: Convert the corrected image to grayscale and apply thresholding again
        gray_rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        _, final_thresh_image = cv2.threshold(gray_rotated_image, 150, 255, cv2.THRESH_BINARY)

        # Convert the corrected image to PIL format for pytesseract
        final_pil_image = Image.fromarray(final_thresh_image)

        # Step 7: Perform OCR using pytesseract with the French language configuration
        custom_config = r'--oem 3 --psm 3 -l fra'
        extracted_text = pytesseract.image_to_string(final_pil_image, config=custom_config)

        return jsonify({'extracted_text': extracted_text})

    except Exception as e:
        logging.error(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
