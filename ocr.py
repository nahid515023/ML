import os
import pytesseract
import cv2
import pandas as pd
from PIL import Image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image_path = "processed_image.png"
    cv2.imwrite(processed_image_path, gray)
    return processed_image_path

def format_text_as_csv(text):
    lines = text.strip().split("\n")
    structured_lines = [line.strip() for line in lines if line.strip()]
    formatted_data = [line.split() for line in structured_lines]
    return formatted_data

def extract_text_to_csv(image_filename, output_csv_filename, lang="eng+ben"):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, image_filename)
        output_csv_path = os.path.join(script_dir, output_csv_filename)
        processed_image_path = preprocess_image(image_path)
        image = Image.open(processed_image_path)
        extracted_text = pytesseract.image_to_string(image, lang=lang)
        formatted_data = format_text_as_csv(extracted_text)
        df = pd.DataFrame(formatted_data)
        df.to_csv(output_csv_path, index=False, header=False)
        print(f"✅ Text extracted and saved to: {output_csv_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    image_filename = "/home/nahid/Desktop/ML/image.jpeg"
    output_csv_filename = "extracted_text.csv"
    extract_text_to_csv(image_filename, output_csv_filename)
