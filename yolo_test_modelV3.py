import cv2
import pytesseract
import pandas as pd
from ultralytics import YOLO

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO(r"C:\Users\ruqai\Desktop\yolo_model1\content\runs\detect\train\weights\best.pt")

# Load image
image_path = r"C:\Users\ruqai\Desktop\image\20241125085600PO_PU420000560_1_page_1.png"
image = cv2.imread(image_path)

class_names = model.names
header_labels = ["Header_PONumber", "Header_PODate", "Header_PaymentTerms", "Header_DeliveryAddress", "Header_Currency"]

# Detect and OCR all relevant boxes
results = model.predict(image_path, conf=0.25)
boxes = []

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label not in header_labels:
            continue

        cropped = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped, config='--psm 6').strip()

        boxes.append({
            "label": label,
            "text": text,
            "x_center": (x1 + x2) // 2,
            "y_center": (y1 + y2) // 2,
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2
        })

#Group into rows based on y_center
boxes.sort(key=lambda b: b["y_center"])
rows = []
row_threshold = 50

for box in boxes:
    placed = False
    for row in rows:
        if abs(row["y_center"] - box["y_center"]) < row_threshold:
            row["boxes"].append(box)
            row["y_center"] = (row["y_center"] + box["y_center"]) // 2
            placed = True
            break
    if not placed:
        rows.append({"y_center": box["y_center"], "boxes": [box]})

# Create a single dictionary for all header fields
header_data = {
    "Header_PONumber": "",
    "Header_PODate": "",
    "Header_PaymentTerms": "",
    "Header_DeliveryAddress": "",
    "Header_Currency": ""
}

# Prioritize left-to-right if useful (optional)
boxes.sort(key=lambda b: b["x_center"])

# Assign detected text to the corresponding header label
for box in boxes:
    if box["label"] in header_data and not header_data[box["label"]]:
        header_data[box["label"]] = box["text"]

#Export to Excel
df = pd.DataFrame([header_data])
column_order = ["Header_PONumber", "Header_PODate", "Header_PaymentTerms", "Header_DeliveryAddress", "Header_Currency"]
df = df[column_order]
df.to_excel("yolo_extracted_Header.xlsx", index=False)
print("Excel file created: yolo_extracted_Header.xlsx")
