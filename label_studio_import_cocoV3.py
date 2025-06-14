# need to convert label studio coco json file first before run the script
# run in powershell
# pip install label-studio converter
# label-studio-converter import coco -i "C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json" -o "C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json"

import os
import json
import logging
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# grab image name in folder
def collect_image_files(folder_path: str) -> Dict[str, str]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    image_map = {}
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        if os.path.isfile(full_path) and ext in IMAGE_EXTENSIONS:
            parts = filename.split('-')
            if len(parts) >= 3:
                key = '-'.join(parts[2:])  # e.g., 20250521165910PO_PU420007433_1_page_1.png
                image_map[key] = filename
    logging.info(f"Collected {len(image_map)} image mappings.")
    return image_map


def load_json_file(json_path: str) -> list:
    """
    Loads and parses a JSON file.

    Returns:
        list: The parsed JSON data.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_image_paths(data: list, image_files: Dict[str, str], project_id: int) -> list:
    """
    Updates the 'image' path in each JSON item using the collected image name

    Returns:
        list: Modified JSON data.
    """
    updated_count = 0
    for item in data:
        image_path = item.get('data', {}).get('image')
        if not image_path:
            continue

        basename = os.path.basename(image_path)
        parts = basename.split('-')
        if len(parts) >= 2:
            trailing_key = '-'.join(parts[1:])
            matched_filename = image_files.get(trailing_key)
            if matched_filename:
                new_path = f"/data/upload/{project_id}/{matched_filename}"
                item['data']['image'] = new_path
                updated_count += 1
                logging.debug(f"Updated: {image_path} → {new_path}")
            else:
                logging.warning(f"No match found for: {image_path}")
    logging.info(f"Updated {updated_count} image paths in JSON.")
    return data


def save_json_file(data: list, output_path: str) -> None:
    """
    Saves the modified JSON data to a file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"JSON updated and saved to: {output_path}")


def main():
    folder_path = r"C:\Users\pc\AppData\Local\label-studio\label-studio\media\upload\24"
    json_file_path = r"C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json"
    project_id = 24

    try:
        image_files = collect_image_files(folder_path)
        json_data = load_json_file(json_file_path)
        updated_data = update_image_paths(json_data, image_files, project_id)

        output_path = json_file_path.replace("result.json", "result_fixed.json")
        save_json_file(updated_data, output_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
