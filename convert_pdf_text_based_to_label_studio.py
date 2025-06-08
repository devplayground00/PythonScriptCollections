import os
import json
from pypdf import PdfReader

# Folder containing PDFs
folder_path = r"C:\Users\Desktop\labelstudio"  # replace this folder path
output_folder = os.path.join(folder_path, "label_studio_json")
os.makedirs(output_folder, exist_ok=True)


def extract_text_from_pdf(file_path):
    extracted_text = []
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted_text.append(page.extract_text() or "")
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

    # Combine all page texts into a single string
    full_text = "\n".join(line.strip() for line in extracted_text if line.strip())
    return full_text


def create_individual_json_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")
            text = extract_text_from_pdf(file_path)
            if text:
                json_data = {
                    "data": {
                        "text": text
                    }
                }

                json_filename = os.path.splitext(filename)[0] + ".json"
                json_output_path = os.path.join(output_folder, json_filename)

                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                print(f"Saved: {json_output_path}")


create_individual_json_files(folder_path)
