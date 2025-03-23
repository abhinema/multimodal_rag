from abc import ABC, abstractmethod
import pytesseract
from pdf2image import convert_from_path
import os

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, input_data):
        pass

class TextExtractor(BaseExtractor):
    def __init__(self, config):
        self.config = config
        if 'tesseract_path' in config and os.path.exists(config['tesseract_path']):
            pytesseract.pytesseract.tesseract_cmd = config['tesseract_path']
        self.language = config.get('language', 'eng')

    def extract(self, pdf_path):
        try:
            pages = convert_from_path(pdf_path)
            text_data = []

            for page_num, page in enumerate(pages):
                text = pytesseract.image_to_string(page, lang=self.language)
                text_data.append({
                    'page': page_num,
                    'text': text,
                    'type': 'text'
                })

            return text_data
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return []