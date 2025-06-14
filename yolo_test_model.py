import cv2
import pytesseract
from ultralytics import YOLO
import os

# Set Tesseract path (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLOv8 model
model = YOLO(r"C:\Users\pc\Desktop\yolo\run\train2\weights\best.pt")

# Image path
image_path = r"C:\Users\pc\Desktop\Image\20250521165928PO_PU420007423_1_page_1.png"
image = cv2.imread(image_path)

# Run detection
results = model.predict(image_path, conf=0.25)
# results = model.predict(
#     image_path,
#     conf=0.05,         # Lower confidence threshold to include weaker detections
#     imgsz=960          # Larger image size improves small object detection
# )


# Class name mapping
class_names = model.names

# Loop through detections
for result in results:
    boxes = result.boxes
    for i, box in enumerate(boxes):
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        print(f"Detected class index: {cls_id} → {label}")

        # Crop the detected region
        cropped = image[y1:y2, x1:x2]

        # Run OCR on the cropped image
        ocr_text = pytesseract.image_to_string(cropped, config='--psm 6')

        print(f"\nClass: {label}")
        print(f"Box: ({x1}, {y1}) → ({x2}, {y2})")
        print(f"OCR Result:\n{ocr_text.strip()}")
