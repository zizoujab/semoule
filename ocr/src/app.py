from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
# Create a Flask app instance
app = Flask(__name__)
def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

# Function to apply perspective transformation and get top-down view
def get_top_down_view(image, doc_contour):
    epsilon = 0.02 * cv2.arcLength(doc_contour, True)
    approx_corners = cv2.approxPolyDP(doc_contour, epsilon, True)

    if len(approx_corners) == 4:
        pts = approx_corners.reshape(4, 2)
        rect = order_points(pts)

        # Determine width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(heightA), int(heightB))

        # Destination points for the perspective transform
        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped
    else:
        return None
# Define a route for the OCR endpoint
@app.route('/ocr', methods=['POST'])
def ocr_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Step 1: Load the image
    # Get the image file from the request
    image_file = request.files['image']

    # Open the image using PIL (Python Imaging Library)
    image = Image.open(io.BytesIO(image_file.read()))


    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 5: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_contour = contours[0]

    # Step 6: Apply perspective transformation to get a top-down view
    warped = get_top_down_view(image, doc_contour)

    if warped is not None:
        # Step 7: Convert the warped image to grayscale and apply adaptive thresholding for enhancement
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Step 8: Perform OCR using Tesseract
        extracted_text = pytesseract.image_to_string(enhanced_image, lang='eng')
        return jsonify({'extracted_text': extracted_text})

    else:
        jsonify({ 'error' : 'Could not detect the document properly.')

# Define the root route to check if the service is running
@app.route('/')
def index():
    return jsonify({'message': 'OCR API is running'}), 200

# Run the app if this file is executed as the main program
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
