# need to convert label studio coco json file first before run the script
# run in powershell
# pip install label-studio converter
# label-studio-converter import coco -i "C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json" -o "C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json"

import os
import json

# Configuration
FOLDER_PATH = r"C:\Users\pc\AppData\Local\label-studio\label-studio\media\upload\24"
JSON_FILE_PATH = r"C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json"
PROJECT_ID = 24
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def build_image_map(folder_path):
    """Builds a mapping of image key → filename based on the trailing pattern in the filename."""
    image_map = {}
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)) and os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
            parts = filename.split('-')
            if len(parts) >= 3:
                key = '-'.join(parts[2:])
                image_map[key] = filename
    return image_map

def update_json_paths(json_data, image_map, project_id):
    """Replaces image paths in JSON data using the image map."""
    for item in json_data:
        if 'data' in item and 'image' in item['data']:
            original_path = item['data']['image']
            basename = os.path.basename(original_path)
            json_parts = basename.split('-')
            if len(json_parts) >= 2:
                key = '-'.join(json_parts[1:])
                if key in image_map:
                    new_filename = image_map[key]
                    new_path = f"/data/upload/{project_id}/{new_filename}"
                    item['data']['image'] = new_path
                    print(f"Updated: {original_path} → {new_path}")
                else:
                    print(f"No match found for: {original_path}")
    return json_data

def main():
    image_map = build_image_map(FOLDER_PATH)

    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = update_json_paths(data, image_map, PROJECT_ID)

    output_path = JSON_FILE_PATH.replace("result.json", "result_fixed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"\nJSON saved to: {output_path}")

if __name__ == "__main__":
    main()
