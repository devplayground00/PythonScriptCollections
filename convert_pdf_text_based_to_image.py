import fitz  # PyMuPDF
import os
# import shutil # optional to create zip files

# Input and output folder setup
input_folder = r"C:\Users\pc\Desktop\Dev playground\SAMPLE\PDF\PO\CELESTICA\CELESTICA PDF 03122024"
output_base_folder = r"C:\Users\pc\Desktop\Training Data\Image"
os.makedirs(output_base_folder, exist_ok=True)

# Dictionary to store output folders for each PDF
output_folders = {}

# Get all PDF files in the input folder
pdf_files = [files for files in os.listdir(input_folder) if files.lower().endswith(".pdf")]

if not pdf_files:
    print("No PDF files found in the input folder.")
    exit()

# Process each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_folder, pdf_file)
    print(f"Processing file: {pdf_file}")

    pdf_name = os.path.splitext(pdf_file)[0]
    output_folder = os.path.join(output_base_folder, pdf_name)
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=600)
        image_path = os.path.join(output_folder, f"{pdf_name}_page_{page_number + 1}.png")
        pix.save(image_path)

    print(f"{pdf_file} converted to images. Saved in {output_folder}")
    output_folders[pdf_name] = output_folder

# optional to create zip files
# # Create a ZIP file for each processed PDF
# for pdf_name, folder in output_folders.items():
#     zip_path = f"./{pdf_name}.zip"
#     shutil.make_archive(zip_path.replace(".zip", ""), 'zip', folder)
#     print(f"ZIP file created: {zip_path}")

print("All processing completed.")
