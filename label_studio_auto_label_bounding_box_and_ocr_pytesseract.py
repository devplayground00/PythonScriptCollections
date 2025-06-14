import os
import json
import pytesseract
from PIL import Image
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np

# Set path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tesseract OCR output levels
LEVELS = {
    'page_num': 1,
    'block_num': 2,
    'par_num': 3,
    'line_num': 4,
    'word_num': 5
}

def create_image_url(filepath):
    filename = os.path.basename(filepath)
    return f'/data/upload/16/{filename}'

def convert_to_ls(image_path, image, tesseract_output, per_level='line_num'):
    image_width, image_height = image.size
    per_level_idx = LEVELS[per_level]
    results = []
    all_scores = []

    num_items = len(tesseract_output['level'])
    # ---------- Draw debug rectangles for each line ----------
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    for i in range(num_items):
        if tesseract_output['level'][i] == LEVELS['line_num']:
            (x, y, w, h) = (
                tesseract_output['left'][i],
                tesseract_output['top'][i],
                tesseract_output['width'][i],
                tesseract_output['height'][i]
            )
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

    debug_folder = Path(image_path).parent / "debug"
    debug_folder.mkdir(exist_ok=True)
    debug_image_path = debug_folder / (Path(image_path).stem + '_debug.png')
    cv2.imwrite(str(debug_image_path), img_cv)
    # ----------------------------------------------------------

    for i in range(num_items):
        if tesseract_output['level'][i] != per_level_idx:
            continue

        # Match hierarchy ID at the current level
        group_id = tesseract_output[per_level][i]

        # Gather all word-level elements that belong to the same group
        words = []
        confidences = []
        for j in range(num_items):
            if tesseract_output[per_level][j] == group_id:
                word = tesseract_output['text'][j].strip()
                if word:
                    words.append(word)
                    conf = tesseract_output['conf'][j]
                    if conf != '-1':
                        try:
                            confidences.append(float(conf) / 100.)
                        except ValueError:
                            pass

        text = ' '.join(words).strip()
        if not text:
            continue

        bbox = {
            'x': 100 * tesseract_output['left'][i] / image_width,
            'y': 100 * tesseract_output['top'][i] / image_height,
            'width': 100 * tesseract_output['width'][i] / image_width,
            'height': 100 * tesseract_output['height'][i] / image_height,
            'rotation': 0
        }

        region_id = str(uuid4())[:10]
        score = sum(confidences) / len(confidences) if confidences else 0

        results.extend([
            {
                'id': region_id,
                'from_name': 'bbox',
                'to_name': 'image',
                'type': 'rectangle',
                'value': bbox
            },
            {
                'id': region_id,
                'from_name': 'transcription',
                'to_name': 'image',
                'type': 'textarea',
                'value': dict(text=[text], **bbox),
                'score': score
            }
        ])
        all_scores.append(score)

    return {
        'data': {
            'ocr': create_image_url(str(image_path))
        },
        'predictions': [{
            'result': results,
            'score': sum(all_scores) / len(all_scores) if all_scores else 0
        }]
    }


def process_images(image_dir_path, per_level='block_num'):
    image_dir = Path(image_dir_path)
    tasks = []

    for f in image_dir.glob('*'):
        if f.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        try:
            with Image.open(f) as image:
                tesseract_output = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                task = convert_to_ls(f, image, tesseract_output, per_level=per_level)
                tasks.append(task)
                print(f"Processed: {f.name}")
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
    return tasks


if __name__ == "__main__":
    output_file = 'ocr_tasks.json'
    image_folder = r'C:\Users\pc\Desktop\Image'
    tasks = process_images(image_folder, per_level='block_num')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    print(f"\nSaved {len(tasks)} tasks to '{output_file}'")
