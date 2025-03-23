import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import os

class TableExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load detection model
        self.processor = AutoImageProcessor.from_pretrained(config['model'])
        self.model = TableTransformerForObjectDetection.from_pretrained(config['model']).to(self.device)

        self.max_size = config.get('max_size', 1000)

    def preprocess_image(self, image):
        # Resize image while maintaining aspect ratio
        w, h = image.size
        scale = min(self.max_size / max(w, h), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))
        return image

    def extract_tables_from_image(self, image):
        """Extract tables from a single image."""
        image = self.preprocess_image(image)

        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.7,
            target_sizes=target_sizes
        )[0]

        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= 0.7:  # Confidence threshold
                # Convert box coordinates to integers
                box = [int(i) for i in box.tolist()]

                # Extract table region
                table_image = image.crop(box)

                # Convert table to dataframe (simplified)
                # In a real implementation, you would use OCR or structure recognition here
                table_data = pd.DataFrame([["Table content"]])

                tables.append({
                    'bbox': box,
                    'confidence': score.item(),
                    'data': table_data
                })

        return tables

    def extract(self, pdf_path):
        """Extract tables from PDF pages."""
        try:
            tables_data = []
            pages = convert_from_path(pdf_path)

            for page_num, page_image in enumerate(pages):
                page_tables = self.extract_tables_from_image(page_image)

                for table_id, table in enumerate(page_tables):
                    tables_data.append({
                        'page': page_num,
                        'table_id': table_id,
                        'bbox': table['bbox'],
                        'confidence': table['confidence'],
                        'data': table['data'],
                        'type': 'table'
                    })

            return tables_data
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []