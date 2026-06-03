"""_summary_
    This file contains the code for extracting text from documents. 
    It uses the CNOCR library to extract text from PDF files to extract text from Word documents. 
    The extracted text is then returned as a string.
"""

import easyocr
from pdf2image import convert_from_path
from typing import Callable
import os




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
                
                text = text.replace("黃", "貳") #fix common misrecognition of "貳" as "黃"
                file.write(f"{confidence:.2f}, {text}\n")  
        print(f"Successfully OCRed from png {input_path} to txt {output_path}")
        return True

    except Exception as e:
        #better error logs 
        print(e)
        return False
    
if __name__ == "__main__":
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)
    
    input_path = "test/input/test.pdf"
    pages = convert_from_path(input_path, dpi=300)
    for i, page in enumerate(pages):
        page.save(f"test/input/page_{i+1}.png", "PNG")
        extract_text_from_image(f"test/input/page_{i+1}.png", f"test/output/page_{i+1}.txt", reader.readtext)
        os.remove(f"test/input/page_{i+1}.png")
        
    