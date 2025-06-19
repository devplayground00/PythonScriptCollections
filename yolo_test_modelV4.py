import cv2
import pytesseract
import pandas as pd
from ultralytics import YOLO

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO(r"C:\Users\ruqai\Desktop\yolo_model1\content\runs\detect\train\weights\best.pt")

# Load image
image_path = r"C:\Users\ruqai\Desktop\yolo\images\val\1dda9e1a-20250521170222PO_PU420007451_1_page_3.png"
image = cv2.imread(image_path)

class_names = model.names
item_labels = ["Item_LineNo", "Item_ItemCode", "Item_BPCatalogno", "Item_BPNeedDate", "Item_Quantity", "Item_Price"]

# Detect and OCR all relevant boxes
results = model.predict(image_path, conf=0.25)
boxes = []

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label not in item_labels:
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

# Group into rows by y_center
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

# Group into structured items
grouped_items = []
current_item = {}

for row in rows:
    row_data = {
        "Item_LineNo": "",
        "Item_ItemCode": "",
        "Item_BPCatalogno": "",
        "Item_BPNeedDate": "",
        "Item_Quantity": "",
        "Item_Price": ""
    }

    row["boxes"].sort(key=lambda b: b["x_center"])
    for box in row["boxes"]:
        if box["label"] in row_data:
            row_data[box["label"]] = box["text"]

    if row_data["Item_LineNo"]:
        current_item = {
            "Item_LineNo": row_data["Item_LineNo"],
            "Item_BPCatalogno": "",
            "Item_ItemCode": "",
            "Deliveries": []
        }
        grouped_items.append(current_item)

    if current_item:
        if row_data["Item_BPCatalogno"]:
            current_item["Item_BPCatalogno"] = row_data["Item_BPCatalogno"]
        if row_data["Item_ItemCode"]:
            current_item["Item_ItemCode"] = row_data["Item_ItemCode"]

        current_item["Deliveries"].append({
            "Item_BPNeedDate": row_data["Item_BPNeedDate"],
            "Item_Quantity": row_data["Item_Quantity"],
            "Item_Price": row_data["Item_Price"]
        })

# Build export rows (repeat item fields only once per item)
export_rows = []
for item in grouped_items:
    for i, delivery in enumerate(item["Deliveries"]):
        export_rows.append({
            "Item_LineNo": item["Item_LineNo"] if i == 0 else "",
            "Item_BPCatalogno": item["Item_BPCatalogno"] if i == 0 else "",
            "Item_ItemCode": item["Item_ItemCode"] if i == 0 else "",
            "Item_BPNeedDate": delivery["Item_BPNeedDate"],
            "Item_Quantity": delivery["Item_Quantity"],
            "Item_Price": delivery["Item_Price"]
        })

# Export to Excel
df = pd.DataFrame(export_rows)
column_order = ["Item_LineNo", "Item_BPCatalogno", "Item_ItemCode", "Item_BPNeedDate", "Item_Quantity", "Item_Price"]
df = df[column_order]
df.to_excel("yolo_extracted_Items_4.xlsx", index=False)
print("Excel file created: yolo_extracted_Items_4.xlsx")
