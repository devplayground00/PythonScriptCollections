# Import required libraries
import os  # For file and path operations
import cv2  # For image loading and cropping
import pytesseract  # For OCR (Optical Character Recognition)
import pandas as pd  # For tabular data export
import logging  # For logging messages to console
from typing import List, Dict  # For type hints
from ultralytics import YOLO  # For loading and using YOLO object detection model

# ---------------- Configuration ----------------
# Paths to required binaries and data
TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
YOLO_MODEL_PATH = r"C:\\Users\\ruqai\\Desktop\\yolo_model1\\content\\runs\\detect\\train\\weights\\best.pt"
IMAGE_FOLDER = r"C:\\Users\\ruqai\\Desktop\\image"
OUTPUT_FILE = "PO_Ichor_V6.xlsx"

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------- Logging ----------------
# Configure logging to display messages in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Constants ----------------
# Define the labels of interest for headers and items
HEADER_LABELS = ["Header_PONumber", "Header_PODate", "Header_PaymentTerms", "Header_DeliveryAddress", "Header_Currency"]
ITEM_LABELS = ["Item_LineNo", "Item_ItemCode", "Item_BPCatalogno", "Item_BPNeedDate", "Item_Quantity", "Item_Price"]
# Y-coordinate threshold to group bounding boxes into rows
ROW_THRESHOLD = 50

# ---------------- Functions ----------------
def load_images(folder: str) -> List[str]:
    # Load and sort PNG files from the folder that match naming pattern
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png") and "_page_" in f])
    logging.info(f"Found {len(files)} image(s).")
    return files

def run_model_on_image(model: YOLO, image_path: str) -> List[Dict]:
    # Run object detection and OCR on an image and return extracted boxes
    image = cv2.imread(image_path)  # Load image
    result = model.predict(image_path, conf=0.25)[0]  # Predict with YOLO
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        cls_id = int(box.cls[0])  # Get class ID
        label = model.names[cls_id]  # Map class ID to label
        cropped = image[y1:y2, x1:x2]  # Crop the bounding box region
        text = pytesseract.image_to_string(cropped, config='--psm 6').strip()  # OCR text
        # Store metadata and extracted text
        boxes.append({"label": label, "text": text, "x_center": (x1 + x2) // 2, "y_center": (y1 + y2) // 2, "x1": x1, "x2": x2, "y1": y1, "y2": y2})
    return boxes

def extract_header_data(boxes: List[Dict]) -> Dict:
    # Extract header data (PO number, date, etc.) from OCR boxes
    header = {label: "" for label in HEADER_LABELS}
    for box in boxes:
        if box["label"] in HEADER_LABELS and not header[box["label"]]:
            header[box["label"]] = box["text"]
    return header

def group_item_boxes(boxes: List[Dict]) -> List[Dict]:
    # Group item fields into rows based on vertical position
    item_boxes = [b for b in boxes if b["label"] in ITEM_LABELS]
    item_boxes.sort(key=lambda b: b["y_center"])  # Sort top to bottom
    rows = []
    for box in item_boxes:
        placed = False
        for row in rows:
            # If box is within vertical threshold of a row, add it
            if abs(row["y_center"] - box["y_center"]) < ROW_THRESHOLD:
                row["boxes"].append(box)
                row["y_center"] = (row["y_center"] + box["y_center"]) // 2
                placed = True
                break
        if not placed:
            # Otherwise start a new row
            rows.append({"y_center": box["y_center"], "boxes": [box]})
    return rows

def process_item_rows(rows: List[Dict]) -> List[Dict]:
    # Convert grouped row boxes into structured item dictionaries
    grouped_items = []
    current_item = {}
    for row in rows:
        row_data = {label: "" for label in ITEM_LABELS}
        row["boxes"].sort(key=lambda b: b["x_center"])  # Sort left to right
        for box in row["boxes"]:
            row_data[box["label"]] = box["text"]

        # New item line starts with LineNo
        if row_data["Item_LineNo"]:
            current_item = {
                "Item_LineNo": row_data["Item_LineNo"],
                "Item_BPCatalogno": "",
                "Item_ItemCode": "",
                "Item_Price": "",
                "list_BPNeedDate_Quantity": []
            }
            grouped_items.append(current_item)

        # Carry forward data for delivery schedule lines
        if current_item:
            if row_data["Item_BPCatalogno"]:
                current_item["Item_BPCatalogno"] = row_data["Item_BPCatalogno"]
            if row_data["Item_ItemCode"]:
                current_item["Item_ItemCode"] = row_data["Item_ItemCode"]
            if row_data["Item_Price"]:
                current_item["Item_Price"] = row_data["Item_Price"]

            current_item["list_BPNeedDate_Quantity"].append({
                "Item_BPNeedDate": row_data["Item_BPNeedDate"],
                "Item_Quantity": row_data["Item_Quantity"]
            })
    return grouped_items

def merge_data(header: Dict, items: List[Dict]) -> List[Dict]:
    # Merge header and item data into final row format for Excel
    merged = []
    for item in items:
        for i, delivery in enumerate(item["list_BPNeedDate_Quantity"]):
            if not (delivery["Item_BPNeedDate"] or delivery["Item_Quantity"]):
                continue
            row = header.copy()
            composite_line_no = f"{item['Item_LineNo']}.{i:02}"
            row.update({
                "Item_LineNo": composite_line_no,
                "Item_BPCatalogno": item["Item_BPCatalogno"],
                "Item_ItemCode": item["Item_ItemCode"],
                "Item_Price": item["Item_Price"],
                "Item_BPNeedDate": delivery["Item_BPNeedDate"],
                "Item_Quantity": delivery["Item_Quantity"]
            })
            merged.append(row)
    return merged

def export_to_excel(data: List[Dict], output_file: str):
    # Convert list of dicts to DataFrame and export to Excel
    df = pd.DataFrame(data)
    column_order = HEADER_LABELS + ["Item_LineNo", "Item_BPCatalogno", "Item_ItemCode", "Item_BPNeedDate", "Item_Quantity", "Item_Price"]
    df = df[column_order]  # Reorder columns
    df.to_excel(output_file, index=False)
    logging.info(f"Exported data to {output_file}")

# ---------------- Main ----------------
def main():
    try:
        model = YOLO(YOLO_MODEL_PATH)  # Load YOLO model
        image_files = load_images(IMAGE_FOLDER)  # Load image filenames
        header_data = {label: "" for label in HEADER_LABELS}  # Initialize header
        all_items = []  # Collect all item rows

        for image_file in image_files:
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            boxes = run_model_on_image(model, image_path)  # Detect + OCR

            # Extract header once from all pages
            header_data.update({k: v for k, v in extract_header_data(boxes).items() if v and not header_data[k]})

            # Extract item rows
            rows = group_item_boxes(boxes)
            items = process_item_rows(rows)
            all_items.extend(items)

        merged_data = merge_data(header_data, all_items)  # Final merge
        export_to_excel(merged_data, OUTPUT_FILE)  # Write output

    except Exception as e:
        logging.error("An error occurred", exc_info=e)  # Log any error

# Execute main script if run as a script
if __name__ == "__main__":
    main()
