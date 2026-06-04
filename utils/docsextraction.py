"""_summary_
    This file contains the code for extracting text from documents. 
    It uses the CNOCR library to extract text from PDF files to extract text from Word documents. 
    The extracted text is then returned as a string.
"""

import easyocr
from pdf2image import convert_from_path
from typing import Callable
import os
import cv2
import re





def extract_text_from_image(input_path: str, output_path: str, ocrapi: Callable[[str], str]) -> bool:
    """
    Extract text from a document using OCR.

    Args:
        input_path (str): The path to the document to be processed.
        output_folder (str): The folder where the extracted text will be saved.

    Returns:
        str: The extracted text.
    """

    # Perform text recognition
    try:
                    
        res = ocrapi(input_path)
        with open(output_path, "w", encoding="utf-8") as file:
            for bounding_box, text, confidence in res:
                
                text = re.sub(r'[a-zA-Z.!\'"^@#%/-{}\[\]]', '', text)
                file.write(f"{text}")  
        print(f"Successfully OCRed from png {input_path} to txt {output_path}")
        return True

    except Exception as e:
        #better error logs 
        print(e)
        return False
def image_preprocessing(input_path: str, output_path: str) -> bool:
    """
    Preprocess the image for better OCR results.

    Args:
        input_path (str): The path to the image to be processed.
        output_folder (str): The folder where the preprocessed image will be saved.

    Returns:
        bool: True if preprocessing was successful, False otherwise.
    """
    try:

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        binary_image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        cv2.imwrite(output_path, binary_image)

    except Exception as e:
        print(e)
        return False 
    
if __name__ == "__main__":
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)
    filename = "EE"
    input_path = f"test/input/{filename}.pdf"
    pages = convert_from_path(input_path, dpi=300)
    for i, page in enumerate(pages):
        page.save(f"test/input/page_{i+1}.png", "PNG")
        extract_text_from_image(f"test/input/page_{i+1}.png", f"test/output/page_{i+1}.txt", reader.readtext)
        os.remove(f"test/input/page_{i+1}.png")
        
        # image_preprocessing(f"test/input/page_{i+1}.png", f"test/input/page_{i+1}_preprocessed.png")
        # os.remove(f"test/input/page_{i+1}.png")
        # extract_text_from_image(f"test/input/page_{i+1}_preprocessed.png", f"test/output/page_{i+1}.txt", reader.readtext)
        # os.remove(f"test/input/page_{i+1}_preprocessed.png")
    cache = []
    #combine all pages into one text file for easier processing by LLM
    for i in range(len(pages)):
        with open(f"test/output/page_{i+1}.txt", "r", encoding="utf-8") as file:
            cache.append(file.read())
        os.remove(f"test/output/page_{i+1}.txt")
    with open(f"test/output/{filename}.txt", "w", encoding="utf-8") as file:
        for text in cache:
            file.write(text)
