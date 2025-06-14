import cv2
import pytesseract
from ultralytics import YOLO

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8 model
model = YOLO('runs/detect/train12/weights/best.pt')

# Image path r"C:\Users\pc\Desktop\yolo\images\val\ed3cf372-20250521165910PO_PU420007433_1_page_1.png"
image_path = r"C:\Users\pc\Desktop\yolo\images\val\ed3cf372-20250521165910PO_PU420007433_1_page_1.png"
# r"C:\Users\pc\Desktop\Image\20250521165928PO_PU420007423_1_page_1.png"

image = cv2.imread(image_path)

# Class name mapping
class_names = model.names

# Define label groups
header_labels = {"Header_PONumber", "Header_PODate", "Header_PaymentTerms", "Header_DeliveryAddress"}
item_labels = {"Item_LineNo", "Item_ItemCode", "Item_BPNeedDate", "Item_Quantity", "Item_Price", "Item_BPCatalogno"}

# --- Step 1: Detect and process HEADER fields ---
print("\n======HEADER FIELDS ======")
results_header = model.predict(image_path, conf=0.25)

for result in results_header:
    boxes = result.boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label not in header_labels:
            continue  # skip non-header

        if (y2 - y1) > 1500:
            continue  # skip large invalid header boxes

        cropped = image[y1:y2, x1:x2]
        ocr_text = pytesseract.image_to_string(cropped, config='--psm 6 -c preserve_interword_spaces=1')

        print(f"\nClass: {label}")
        print(f"Box: ({x1}, {y1}) → ({x2}, {y2})")
        print(f"OCR Result:\n{ocr_text.strip()}")

# --- Step 2: Detect and process ITEM fields ---
print("\n======ITEM FIELDS ======")
results_item = model.predict(image_path, conf=0.25)

for result in results_item:
    boxes = result.boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label not in item_labels:
            continue  # skip non-item

        cropped = image[y1:y2, x1:x2]
        ocr_text = pytesseract.image_to_string(cropped, config='--psm 6 -c preserve_interword_spaces=1')

        print(f"\nClass: {label}")
        print(f"Box: ({x1}, {y1}) → ({x2}, {y2})")
        print(f"OCR Result:\n{ocr_text.strip()}")
