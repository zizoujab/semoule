import os
import pprint
import requests
import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import io
import cv2
import numpy as np
import logging
from dotenv import load_dotenv
import json
import traceback
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

app = Flask(__name__)

from transformers import AutoModelForTokenClassification
from transformers import AutoProcessor


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


@app.route('/ml', methods=['POST'])
def layout_lm():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:

        # Step 1: Load the image file from the request
        image_file = request.files['image']
        filename = request.form['filename']
        print(filename)

        task_id = create_task(image_file, filename)

        image = Image.open(image_file).convert("RGB")
        output_path = f"../out/{filename}"
        print("Image width:", image.width)
        print("Image height:", image.height)
        annotated_image = task_inference(task_id, image.width, image.height, filename, image)
        return jsonify({'file_uri': '/annotated/' + filename, 'file_name' : filename})
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)  # shows full traceback
        return jsonify({'error': str(e)}), 500


@app.route('/annotated/<filename>', methods=['GET'])
def get_annotated_image(filename):
    try:
        image_path = os.path.join('../out', filename)

        if not os.path.isfile(image_path):
            return jsonify({"error": "Image not found"}), 404

        return send_from_directory('../out', filename)

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)  # shows full traceback
        return jsonify({'error': str(e)}), 500


def task_inference(task_id, image_width, image_height, filename, image):
    ocr_words = get_bboxes(task_id, image_width, image_height)

    # normalize data :
    invoice_data = {
        "id": task_id,
        "file_name": filename,
        "tokens": [],
        "bboxes": [],
        "ner_tags": []
    }
    for ocr_item in ocr_words:
        w = ocr_item['width']
        h = ocr_item['height']
        x_word = ocr_item['x']
        y_word = ocr_item['y']
        x1_word = ocr_item['x'] + w
        y1_word = ocr_item['y'] + h
        invoice_data['tokens'].append(ocr_item['text'][0])
        if x_word > 100:
            print(f'x_word > 100 : {x_word} {task_id}')
        if y_word > 100:
            print(f'x_word :  > 100{y_word} {task_id}')
        invoice_data['bboxes'].append(normalize_box([x_word, y_word, x1_word, y1_word], w, h))
        invoice_data['ner_tags'].append(0)

    # model = AutoModelForTokenClassification.from_pretrained("../model/checkpoint-500")
    # processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    processor = LayoutLMv3Processor.from_pretrained("../model/checkpoint-500")
    model = LayoutLMv3ForTokenClassification.from_pretrained("../model/checkpoint-500")

    print(type(processor))

    words = invoice_data["tokens"]
    boxes = invoice_data["bboxes"]
    encoding = processor(image, words, boxes=boxes, return_tensors="pt")
    for k, v in encoding.items():
        print(k, v.shape)

    with torch.no_grad():
        outputs = model(**encoding)
    print('just before logits')

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    # labels = encoding.labels.squeeze().tolist()
    # print('labels are :')
    # print(labels)

    token_boxes = encoding.bbox.squeeze().tolist()
    width, height = image.size

    true_predictions = [model.config.id2label[pred] for pred in predictions]
    # true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    # label2color = {'invoice_no': 'blue', 'date': 'green', 'amount': 'orange'}

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline='green')
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill='green', font=font)

    output_path = f"../out/{filename}"
    image.save(output_path)
    print(f"Annotated image saved to: {output_path}")

    return image


def get_bboxes(task_id: str, original_width: str, original_height: str):
    api_key = os.getenv("API_KEY")
    label_studio_url = os.getenv("LABEL_STUDIO_URL")
    ml_id = os.getenv("ML_ID")
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    ml_url = f"{label_studio_url}/api/ml/{ml_id}/interactive-annotating"
    payload = {
        "task": task_id,
        "context": {
            "result": [
                {
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "x": 0,
                        "y": 0,
                        "width": 100,
                        "height": 100,
                        "rotation": 0,
                        "rectanglelabels": ["api"]
                    },
                    "from_name": "bbox",
                    "to_name": "image",
                    "type": "api",
                    "origin": "api",
                    "include_bbox": True
                }
            ]
        }
    }
    response = requests.post(ml_url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result['data']['result'][0]['value']['bboxes']
    else:
        print(f"Failed to get task text : {response.status_code}, {response.text}")


def create_task(file, filename):
    api_key = os.getenv("API_KEY")
    label_studio_url = os.getenv("LABEL_STUDIO_URL")
    project_id = os.getenv("PROJECT_ID")
    headers = {
        "Authorization": f"Token {api_key}"
    }

    print(file.mimetype)
    print(file.filename)
    files = {'file': (filename, file.stream, file.mimetype)}
    response = requests.post(
        f'{label_studio_url}/api/projects/{project_id}/import?commit_to_project=true&return_task_ids=true',
        headers=headers,
        files=files
    )
    if response.status_code == 201:
        response_json = response.json()
        print(response_json)
        task_id = response_json['task_ids'][0]
        print(f"[+] Task created with ID: {task_id}")
        return task_id
    else:
        print(f"Failed to import image: {response.status_code}, {response.text}")


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def normalize_box(box, width, height):
    # we are getting the coodrinates in percentage value (x/w * 100) not in pixel values sy tesseract.py in label studio project
    return [
        int(10 * box[0]),
        int(10 * box[1]),
        int(10 * box[2]),
        int(10 * box[3]),
    ]


def iob_to_label(label):
    label = label
    if not label:
        return 'X'
    return label


if __name__ == '__main__':
    load_dotenv()
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
from flask import Flask, request, jsonify

app = Flask(__name__)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
