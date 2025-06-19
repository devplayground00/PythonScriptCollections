import os
import cv2
import pytesseract
import pandas as pd
from ultralytics import YOLO

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO(r"C:\Users\ruqai\Desktop\yolo_model1\content\runs\detect\train\weights\best.pt")

# folder path
image_folder = r"C:\Users\ruqai\Desktop\image"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png") and "_page_" in f])

# Label config
class_names = model.names
header_labels = ["Header_PONumber", "Header_PODate", "Header_PaymentTerms", "Header_DeliveryAddress", "Header_Currency"]
item_labels = ["Item_LineNo", "Item_ItemCode", "Item_BPCatalogno", "Item_BPNeedDate", "Item_Quantity", "Item_Price"]

# Result containers
header_data = {label: "" for label in header_labels}
grouped_items = []

# Go through each image file
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    results = model.predict(image_path, conf=0.25)
    boxes = []

    # Collect all YOLO + OCR boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            cropped = image[y1:y2, x1:x2]
            text = pytesseract.image_to_string(cropped, config='--psm 6').strip()

            boxes.append({
                "label": label,
                "text": text,
                "x_center": (x1 + x2) // 2,
                "y_center": (y1 + y2) // 2,
                "x1": x1, "x2": x2, "y1": y1, "y2": y2
            })

    # ---------------- HEADER ----------------
    for box in boxes:
        if box["label"] in header_labels and not header_data[box["label"]]:
            header_data[box["label"]] = box["text"]

    # ---------------- ITEMS ----------------
    item_boxes = [box for box in boxes if box["label"] in item_labels]
    item_boxes.sort(key=lambda b: b["y_center"])
    rows = []
    row_threshold = 50

    for box in item_boxes:
        placed = False
        for row in rows:
            if abs(row["y_center"] - box["y_center"]) < row_threshold:
                row["boxes"].append(box)
                row["y_center"] = (row["y_center"] + box["y_center"]) // 2
                placed = True
                break
        if not placed:
            rows.append({"y_center": box["y_center"], "boxes": [box]})


    current_item = {}
    for row in rows:
        row_data = {label: "" for label in item_labels}
        row["boxes"].sort(key=lambda b: b["x_center"])
        for box in row["boxes"]:
            if box["label"] in row_data:
                row_data[box["label"]] = box["text"]

        # group based on line_no
        if row_data["Item_LineNo"]:
            current_item = {
                "Item_LineNo": row_data["Item_LineNo"],
                "Item_BPCatalogno": "",
                "Item_ItemCode": "",
                "Item_Price":"",
                "list_BPNeedDate_Quantity": []
            }
            grouped_items.append(current_item)

        # Carry forward ItemCode and BPCatalogno
        if current_item:
            if row_data["Item_BPCatalogno"]:
                current_item["Item_BPCatalogno"] = row_data["Item_BPCatalogno"]
            if row_data["Item_ItemCode"]:
                current_item["Item_ItemCode"] = row_data["Item_ItemCode"]
            if row_data["Item_Price"]:
                current_item["Item_Price"] = row_data["Item_Price"]

            current_item["list_BPNeedDate_Quantity"].append({
                "Item_BPNeedDate": row_data["Item_BPNeedDate"],
                "Item_Quantity": row_data["Item_Quantity"],
            })

# MERGE HEADER + ITEMS
combined_rows = []
for item in grouped_items:
    list_BPNeedDate_Quantity = item["list_BPNeedDate_Quantity"]

    # Find the first delivery with
    lifted_index = None
    for idx, d in enumerate(list_BPNeedDate_Quantity):
        if d["Item_BPNeedDate"] or d["Item_Quantity"]:
            lifted_index = idx
            break

    for i, delivery in enumerate(list_BPNeedDate_Quantity):
        # Skip completely empty rows
        if not (delivery["Item_BPNeedDate"] or delivery["Item_Quantity"]):
            continue

        row = {**header_data}

        main_line_no = item["Item_LineNo"]
        sub_index = i
        composite_line_no = f"{main_line_no}.{sub_index:02}"

        row.update({
            "Item_LineNo": composite_line_no,
            "Item_BPCatalogno": item["Item_BPCatalogno"],
            "Item_ItemCode": item["Item_ItemCode"],
            "Item_Price": item["Item_Price"],
            "Item_BPNeedDate": delivery["Item_BPNeedDate"],
            "Item_Quantity": delivery["Item_Quantity"],
        })

        combined_rows.append(row)

# ---------------- EXPORT ----------------
df = pd.DataFrame(combined_rows)
column_order = [
    "Header_PONumber", "Header_PODate", "Header_PaymentTerms", "Header_DeliveryAddress", "Header_Currency",
    "Item_LineNo", "Item_BPCatalogno", "Item_ItemCode", "Item_BPNeedDate", "Item_Quantity", "Item_Price"
]
df = df[column_order]
df.to_excel("PO_Ichor_V5.xlsx", index=False)
print("Excel created: PO_Ichor_V5.xlsx")
