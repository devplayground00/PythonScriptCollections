# need to convert label studio coco json file first before run the script
# run in powershell
# pip install label-studio converter
# label-studio-converter import coco -i "C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json" -o "C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json"

import os
import json

# Folder and file paths
folder_path = r"C:\Users\pc\AppData\Local\label-studio\label-studio\media\upload\24"
json_file_path = r"C:\Users\pc\Downloads\project-18-at-2025-06-14-23-47-38908621\result.json"
project_id = 24

# Allowed image extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Get list of image files with their key matching part
image_files = {}
for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in image_extensions:
        # Extract the trailing part after the second dash `-`, e.g. 20250521165910PO_PU420007433_1_page_1.png
        parts = filename.split('-')
        if len(parts) >= 3:
            key = '-'.join(parts[2:])  # trailing part
            image_files[key] = filename

# Load JSON
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process and replace paths in JSON
for item in data:
    if 'data' in item and 'image' in item['data']:
        original_path = item['data']['image']
        basename = os.path.basename(original_path)

        # Extract the matching key from JSON path
        json_parts = basename.split('-')
        if len(json_parts) >= 2:
            trailing_key = '-'.join(json_parts[1:])  # e.g. 20250521165834PO_PU420007442_1_page_1.png

            # Match and replace
            if trailing_key in image_files:
                matched_filename = image_files[trailing_key]
                new_path = f"/data/upload/{project_id}/{matched_filename}"
                item['data']['image'] = new_path
                print(f"Updated: {original_path} → {new_path}")
            else:
                print(f"No match found for: {original_path}")

# Save updated JSON
output_path = json_file_path.replace("result.json", "result_fixed.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON updated and saved to: {output_path}")
