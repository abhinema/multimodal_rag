from .base_extractor import BaseExtractor
import fitz  # PyMuPDF
import re
import uuid
import os

class TextExtractor(BaseExtractor):
    def __init__(self, config):
        self.config = config
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        
    def extract(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text_data = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Skip empty pages
                if not text.strip():
                    continue
                
                # Chunk the text
                chunks = self._chunk_text(text, page_num)
                text_data.extend(chunks)
                
            return text_data
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return []
    
    def _chunk_text(self, text, page_num):
        """Split text into chunks with overlap."""
        chunks = []
        
        # Clean text
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # If text is shorter than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [{
                'id': str(uuid.uuid4()),
                'page': page_num,
                'text': text,
                'type': 'text'
            }]
        
        # Split text into chunks
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Try to find a period, question mark, or exclamation point
                for i in range(end-1, max(start, end-100), -1):
                    if text[i] in ['.', '!', '?', '\\n'] and text[i+1:i+2].isspace():
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'page': page_num,
                    'text': chunk,
                    'type': 'text'
                })
            
            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap
            
        return chunks