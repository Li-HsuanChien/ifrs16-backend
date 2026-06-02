"""_summary_
    This file contains the code for extracting text from documents. 
    It uses the CNOCR library to extract text from PDF files to extract text from Word documents. 
    The extracted text is then returned as a string.
"""

import easyocr
from pdf2image import convert_from_path



def extract_text_from_document(input_path: str, output_folder: str) -> str:
    """
    Extract text from a document using OCR.

    Args:
        input_path (str): The path to the document to be processed.
        output_folder (str): The folder where the extracted text will be saved.

    Returns:
        str: The extracted text.
    """
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)

    # Perform text recognition
    pages = convert_from_path(input_path, dpi=300)

    for i, page in enumerate(pages):
        page.save(f"test/input/page_{i+1}.png", "PNG")
        res = reader.readtext(f"test/input/page_{i+1}.png")
        for bounding_box, text, confidence in res:
            print(f"Text: {text} (Confidence: {confidence:.2f})")
        
        # with open(f"{output_folder}page_{i+1}.txt", "w", encoding="utf-8") as file:
        #     file.write(extracted_text)
    
    # return extracted_text

if __name__ == "__main__":
    extract_text_from_document("test/input/test.pdf", "test/output/")