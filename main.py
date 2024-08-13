import fitz  # PyMuPDF
import cv2
import pytesseract
import os
import pandas as pd
import numpy as np
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# teseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# os.environ["teseract_path"] = teseract_path


def parse_pdf(pdf_path):
    # Step 1: Parse the PDF
    pdf_document = fitz.open(pdf_path)
    return pdf_document

def extract_images_from_pdf(pdf_document):
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        images.append(img)
    return images

def detect_tables_in_images(images):
    tables = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        inverted_binary = 255 - binary
        
        contours, _ = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        table_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        tables.append(table_contours)
    return tables

def extract_text_from_image(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    table_image = image[y:y+h, x:x+w]
    text = pytesseract.image_to_string(table_image, config='--psm 6')
    return text

def clean_and_format_data(raw_data):
    # Example cleaning: splitting rows and columns
    rows = raw_data.strip().split('\n')
    data = [row.split() for row in rows]
    return pd.DataFrame(data)

def extract_tables_from_pdf(pdf_path):
    pdf_document = parse_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_document)
    table_contours = detect_tables_in_images(images)
    
    tables = []
    for img, contours in zip(images, table_contours):
        for contour in contours:
            raw_text = extract_text_from_image(img, contour)
            cleaned_data = clean_and_format_data(raw_text)
            tables.append(cleaned_data)
    
    return tables

def export_to_csv(tables, output_path):
    combined_df = pd.concat(tables, ignore_index=True)
    combined_df.to_csv(output_path, index=False)

# Example usage
pdf_path = 'new.pdf'
output_csv_path = 'output.csv'
tables = extract_tables_from_pdf(pdf_path)
export_to_csv(tables, output_csv_path)