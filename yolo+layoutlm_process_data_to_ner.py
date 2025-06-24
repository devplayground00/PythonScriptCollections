# pip install transformers datasets torchvision


# ---------------- Imports ----------------
import os  # For file and path operations
import cv2  # For image loading and cropping
import pytesseract  # For OCR (Optical Character Recognition)
import pandas as pd  # For tabular data export
import logging  # For logging messages to console
import json  # For saving training data
from typing import List, Dict  # For type hints
from ultralytics import YOLO  # For loading and using YOLO object detection model

# ---------------- Configuration ----------------
# Paths to required binaries and data
TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
YOLO_MODEL_PATH = r"C:\Users\ruqai\Desktop\ML PO\best.pt"
IMAGE_FOLDER = r"C:\Users\ruqai\Desktop\image"
OUTPUT_FILE = r"C:\Users\ruqai\Desktop\PO_Ichor_V6.xlsx"
LAYOUTLMV3_OUTPUT_JSON = r"C:\Users\ruqai\Desktop\layoutlm.json"

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Constants ----------------
HEADER_LABELS = ["Header_PONumber", "Header_PODate", "Header_PaymentTerms", "Header_DeliveryAddress", "Header_Currency"]
ITEM_LABELS = ["Item_LineNo", "Item_ItemCode", "Item_BPCatalogno", "Item_BPNeedDate", "Item_Quantity", "Item_Price"]
ROW_THRESHOLD = 50

# ---------------- Core Functions ----------------
def load_images(folder: str) -> List[str]:
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png") and "_page_" in f])
    logging.info(f"Found {len(files)} image(s).")
    return files

def run_model_on_image(model: YOLO, image_path: str) -> List[Dict]:
    image = cv2.imread(image_path)
    result = model.predict(image_path, conf=0.25)[0]
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        cropped = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped, config='--psm 6').strip()
        boxes.append({
            "label": label,
            "text": text,
            "x_center": (x1 + x2) // 2,
            "y_center": (y1 + y2) // 2,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    return boxes

def extract_header_data(boxes: List[Dict]) -> Dict:
    header = {label: "" for label in HEADER_LABELS}
    for box in boxes:
        if box["label"] in HEADER_LABELS and not header[box["label"]]:
            header[box["label"]] = box["text"]
    return header

def group_item_boxes(boxes: List[Dict]) -> List[Dict]:
    item_boxes = [b for b in boxes if b["label"] in ITEM_LABELS]
    item_boxes.sort(key=lambda b: b["y_center"])
    rows = []
    for box in item_boxes:
        placed = False
        for row in rows:
            if abs(row["y_center"] - box["y_center"]) < ROW_THRESHOLD:
                row["boxes"].append(box)
                row["y_center"] = (row["y_center"] + box["y_center"]) // 2
                placed = True
                break
        if not placed:
            rows.append({"y_center": box["y_center"], "boxes": [box]})
    return rows

def process_item_rows(rows: List[Dict]) -> List[Dict]:
    grouped_items = []
    current_item = {}
    for row in rows:
        row_data = {label: "" for label in ITEM_LABELS}
        row["boxes"].sort(key=lambda b: b["x_center"])
        for box in row["boxes"]:
            row_data[box["label"]] = box["text"]
        if row_data["Item_LineNo"]:
            current_item = {
                "Item_LineNo": row_data["Item_LineNo"],
                "Item_BPCatalogno": "",
                "Item_ItemCode": "",
                "Item_Price": "",
                "list_BPNeedDate_Quantity": []
            }
            grouped_items.append(current_item)
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
    merged = []
    for item in items:
        for i, delivery in enumerate(item["list_BPNeedDate_Quantity"]):
            if not (delivery["Item_BPNeedDate"] or delivery["Item_Quantity"]):
                continue
            row = header.copy()
            row.update({
                "Item_LineNo": f"{item['Item_LineNo']}.{i:02}",
                "Item_BPCatalogno": item["Item_BPCatalogno"],
                "Item_ItemCode": item["Item_ItemCode"],
                "Item_Price": item["Item_Price"],
                "Item_BPNeedDate": delivery["Item_BPNeedDate"],
                "Item_Quantity": delivery["Item_Quantity"]
            })
            merged.append(row)
    return merged

def export_to_excel(data: List[Dict], output_file: str):
    df = pd.DataFrame(data)
    column_order = HEADER_LABELS + ["Item_LineNo", "Item_BPCatalogno", "Item_ItemCode", "Item_BPNeedDate", "Item_Quantity", "Item_Price"]
    df = df[column_order]
    df.to_excel(output_file, index=False)
    logging.info(f"Exported data to {output_file}")

# ----------------  LayoutLMv3 Dataset Export ----------------
def normalize_box(box, width, height):
    return [
        int((box["x1"] / width) * 1000),
        int((box["y1"] / height) * 1000),
        int((box["x2"] / width) * 1000),
        int((box["y2"] / height) * 1000),
    ]

def generate_layoutlmv3_training_data(output_json_path=LAYOUTLMV3_OUTPUT_JSON):
    try:
        model = YOLO(YOLO_MODEL_PATH)
        image_files = load_images(IMAGE_FOLDER)
        samples = []

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            boxes = run_model_on_image(model, image_path)

            words, bboxes, ner_tags = [], [], []

            for box in boxes:
                text = box["text"].strip()
                if not text:
                    continue
                words.append(text)
                bboxes.append(normalize_box(box, width, height))
                if box["label"] in HEADER_LABELS + ITEM_LABELS:
                    ner_tags.append(f"B-{box['label']}")
                else:
                    ner_tags.append("O")

            samples.append({
                "id": f"doc_{idx:04}",
                "words": words,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image": image_path
            })

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        logging.info(f"Saved {len(samples)} LayoutLMv3 samples to {output_json_path}")
    except Exception as e:
        logging.error("Failed to generate LayoutLMv3 training data", exc_info=e)

# ---------------- Main ----------------
def main():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        image_files = load_images(IMAGE_FOLDER)
        header_data = {label: "" for label in HEADER_LABELS}
        all_items = []

        for image_file in image_files:
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            boxes = run_model_on_image(model, image_path)
            header_data.update({k: v for k, v in extract_header_data(boxes).items() if v and not header_data[k]})
            rows = group_item_boxes(boxes)
            items = process_item_rows(rows)
            all_items.extend(items)

        merged_data = merge_data(header_data, all_items)
        export_to_excel(merged_data, OUTPUT_FILE)
    except Exception as e:
        logging.error("An error occurred", exc_info=e)

# ---------------- Mode Switch ----------------
if __name__ == "__main__":
    mode = "train_dataset"
    if mode == "export":
        main()
    elif mode == "train_dataset":
        generate_layoutlmv3_training_data()
