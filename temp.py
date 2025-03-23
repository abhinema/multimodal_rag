# create_project_structure.py
import os
import shutil
import zipfile

def create_project_structure():
    # Define the base directory
    base_dir = "multimodal_rag"

    # Create main project directory
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    # Create directory structure
    directories = [
        "config",
        "src/extractors",
        "src/embedders",
        "src/vectordb",
        "src/rag",
        "src/api",
        "ui/static",
        "ui/templates",
        "temp",  # For temporary file storage
    ]

    for dir_path in directories:
        os.makedirs(os.path.join(base_dir, dir_path))

    # Create requirements.txt
    requirements = """fastapi==0.68.1
uvicorn==0.15.0
python-multipart==0.0.5
pytesseract==0.3.8
pdf2image==1.16.0
pandas==1.3.3
torch==1.9.0
transformers==4.30.2
sentence-transformers==2.1.0
pymilvus==2.0.2
pyyaml==5.4.1
python-dotenv==0.19.0
numpy==1.21.0
Pillow==8.3.1
"""

    with open(os.path.join(base_dir, "requirements.txt"), "w") as f:
        f.write(requirements)

    # Create config.yaml
    config = """extractors:
  text:
    model: "tesseract"
    language: "eng"
    tesseract_path: "/usr/bin/tesseract"  # Update for your system
  table:
    model: "microsoft/table-transformer-detection"
    structure_model: "microsoft/table-transformer-structure-recognition"
    max_size: 1000
  image:
    model: "openai/clip-vit-base-patch32"
    confidence_threshold: 0.5

embedders:
  text:
    model: "sentence-transformers/all-mpnet-base-v2"
    dimension: 768
  table:
    model: "microsoft/tapas-base"
    dimension: 768
  image:
    model: "openai/clip-vit-base-patch32"
    dimension: 512

vector_store:
  type: "milvus"
  host: "localhost"
  port: 19530
  collection_prefix: "rag_"

api:
  host: "0.0.0.0"
  port: 8000
  debug: true
"""

    with open(os.path.join(base_dir, "config", "config.yaml"), "w") as f:
        f.write(config)

    # Create Python files
    files = {
        # Base extractor
        "src/extractors/__init__.py": """from .text_extractor import TextExtractor
from .table_extractor import TableExtractor
from .image_extractor import ImageExtractor
""",

        # Text extractor
        "src/extractors/text_extractor.py": """from abc import ABC, abstractmethod
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
""",

        # Table extractor
        "src/extractors/table_extractor.py": """import torch
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
""",

        # Image extractor
        "src/extractors/image_extractor.py": """from PIL import Image
import torch
from pdf2image import convert_from_path
import numpy as np
import os

class ImageExtractor:
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)

    def extract_images_from_page(self, page_image, page_num):
        """Extract images from a single page (simplified)."""
        # In a real implementation, you would use an object detection model
        # Here we're just treating the whole page as an image

        # Save the page image to a temporary file
        temp_path = f"temp/page_{page_num}.jpg"
        page_image.save(temp_path)

        return [{
            'page': page_num,
            'image_id': 0,
            'path': temp_path,
            'type': 'image'
        }]

    def extract(self, pdf_path):
        """Extract images from PDF pages."""
        try:
            images_data = []
            pages = convert_from_path(pdf_path)

            for page_num, page_image in enumerate(pages):
                page_images = self.extract_images_from_page(page_image, page_num)
                images_data.extend(page_images)

            return images_data
        except Exception as e:
            print(f"Error extracting images: {str(e)}")
            return []
""",

        # Embedders
        "src/embedders/__init__.py": """from .text_embedder import TextEmbedder
from .table_embedder import TableEmbedder
from .image_embedder import ImageEmbedder
""",

        # Text embedder
        "src/embedders/text_embedder.py": """from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config['model'])
        self.dimension = config['dimension']

    def embed(self, text_data):
        """Generate embeddings for text data."""
        try:
            embeddings = []

            for item in text_data:
                # Generate embedding
                embedding = self.model.encode(item['text'])

                # Ensure correct dimension
                if len(embedding) != self.dimension:
                    embedding = np.resize(embedding, (self.dimension,))

                embeddings.append({
                    'embedding': embedding,
                    'metadata': item
                })

            return embeddings
        except Exception as e:
            print(f"Error embedding text: {str(e)}")
            return []
""",

        # Table embedder
        "src/embedders/table_embedder.py": """import torch
import numpy as np
import pandas as pd

class TableEmbedder:
    def __init__(self, config):
        self.config = config
        self.dimension = config['dimension']

        # In a real implementation, you would load a table embedding model
        # For simplicity, we'll use a placeholder approach

    def table_to_text(self, df):
        """Convert table to text representation."""
        text = ""
        for i, row in df.iterrows():
            text += " | ".join(str(cell) for cell in row) + "\n"
        return text

    def embed(self, table_data):
        """Generate embeddings for tables."""
        try:
            embeddings = []

            for table in table_data:
                # Convert table to text
                table_text = self.table_to_text(table['data'])

                # Generate a random embedding (placeholder)
                # In a real implementation, you would use a proper embedding model
                embedding = np.random.randn(self.dimension).astype(np.float32)

                embeddings.append({
                    'embedding': embedding,
                    'metadata': {
                        'page': table['page'],
                        'table_id': table['table_id'],
                        'bbox': table['bbox'],
                        'type': 'table'
                    }
                })

            return embeddings
        except Exception as e:
            print(f"Error embedding table: {str(e)}")
            return []
""",

        # Image embedder
        "src/embedders/image_embedder.py": """import torch
from PIL import Image
import numpy as np
import os

class ImageEmbedder:
    def __init__(self, config):
        self.config = config
        self.dimension = config['dimension']

        # In a real implementation, you would load CLIP or another image embedding model
        # For simplicity, we'll use a placeholder approach

    def embed(self, image_data):
        """Generate embeddings for images."""
        try:
            embeddings = []

            for item in image_data:
                # Generate a random embedding (placeholder)
                # In a real implementation, you would use a proper embedding model
                embedding = np.random.randn(self.dimension).astype(np.float32)

                embeddings.append({
                    'embedding': embedding,
                    'metadata': item
                })

            return embeddings
        except Exception as e:
            print(f"Error embedding image: {str(e)}")
            return []
""",

        # Vector DB
        "src/vectordb/__init__.py": "",

        # Vector store
        "src/vectordb/vector_store.py": """import numpy as np
import os
import json
from datetime import datetime

class VectorStore:
    def __init__(self, config):
        self.config = config
        self.collection_prefix = config.get('collection_prefix', 'rag_')

        # In a real implementation, you would connect to Milvus or another vector DB
        # For simplicity, we'll use a file-based approach
        self.collections = {}

        # Create temp directory for vector storage
        os.makedirs('temp/vectors', exist_ok=True)

    def create_collection(self, name, dimension):
        """Create a new collection."""
        collection_name = f"{self.collection_prefix}{name}"

        if collection_name not in self.collections:
            self.collections[collection_name] = {
                'name': collection_name,
                'dimension': dimension,
                'vectors': [],
                'metadata': []
            }

            print(f"Created collection: {collection_name}")

        return collection_name

    def insert(self, collection_name, embeddings, metadata=None):
        """Insert vectors into collection."""
        collection_name = f"{self.collection_prefix}{collection_name}"

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        if metadata is None:
            metadata = [{}] * len(embeddings)

        # Add vectors and metadata
        for i, (vector, meta) in enumerate(zip(embeddings, metadata)):
            self.collections[collection_name]['vectors'].append(vector)
            self.collections[collection_name]['metadata'].append(meta)

        # Save to file
        self._save_collection(collection_name)

        print(f"Inserted {len(embeddings)} vectors into {collection_name}")

    def search(self, collection_name, query_vector, top_k=5):
        """Search for similar vectors."""
        collection_name = f"{self.collection_prefix}{collection_name}"

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")

        collection = self.collections[collection_name]

        if not collection['vectors']:
            return []

        # Convert to numpy array
        vectors = np.array(collection['vectors'])
        query_vector = np.array(query_vector)

        # Compute cosine similarity
        similarities = np.dot(vectors, query_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'score': float(similarities[idx]),
                'metadata': collection['metadata'][idx]
            })

        return results

    def _save_collection(self, collection_name):
        """Save collection to file."""
        collection = self.collections[collection_name]

        # Save vectors
        vectors_file = f"temp/vectors/{collection_name}_vectors.npy"
        np.save(vectors_file, np.array(collection['vectors']))

        # Save metadata
        metadata_file = f"temp/vectors/{collection_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(collection['metadata'], f)

    def _load_collection(self, collection_name):
        """Load collection from file."""
        vectors_file = f"temp/vectors/{collection_name}_vectors.npy"
        metadata_file = f"temp/vectors/{collection_name}_metadata.json"

        if os.path.exists(vectors_file) and os.path.exists(metadata_file):
            # Load vectors
            vectors = np.load(vectors_file).tolist()

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.collections[collection_name] = {
                'name': collection_name,
                'dimension': len(vectors[0]) if vectors else 0,
                'vectors': vectors,
                'metadata': metadata
            }

            print(f"Loaded collection: {collection_name}")
""",

        # RAG
        "src/rag/__init__.py": "",

        # Retriever
        "src/rag/retriever.py": """from ..extractors.text_extractor import TextExtractor
from ..extractors.table_extractor import TableExtractor
from ..extractors.image_extractor import ImageExtractor
from ..embedders.text_embedder import TextEmbedder
from ..embedders.table_embedder import TableEmbedder
from ..embedders.image_embedder import ImageEmbedder
from ..vectordb.vector_store import VectorStore
import os

class MultimodalRAG:
    def __init__(self, config):
        # Initialize extractors
        self.text_extractor = TextExtractor(config['extractors']['text'])
        self.table_extractor = TableExtractor(config['extractors']['table'])
        self.image_extractor = ImageExtractor(config['extractors']['image'])

        # Initialize embedders
        self.text_embedder = TextEmbedder(config['embedders']['text'])
        self.table_embedder = TableEmbedder(config['embedders']['table'])
        self.image_embedder = ImageEmbedder(config['embedders']['image'])

        # Initialize vector store
        self.vector_store = VectorStore(config['vector_store'])

        # Create collections
        self.vector_store.create_collection('text', config['embedders']['text']['dimension'])
        self.vector_store.create_collection('table', config['embedders']['table']['dimension'])
        self.vector_store.create_collection('image', config['embedders']['image']['dimension'])

    def process_document(self, pdf_path):
        """Process a PDF document and store its embeddings."""
        try:
            # Create temp directory if it doesn't exist
            os.makedirs('temp', exist_ok=True)

            # Extract content
            print(f"Extracting text from {pdf_path}...")
            text_data = self.text_extractor.extract(pdf_path)
            print(f"Extracted {len(text_data)} text segments")

            print(f"Extracting tables from {pdf_path}...")
            table_data = self.table_extractor.extract(pdf_path)
            print(f"Extracted {len(table_data)} tables")

            print(f"Extracting images from {pdf_path}...")
            image_data = self.image_extractor.extract(pdf_path)
            print(f"Extracted {len(image_data)} images")

            # Generate embeddings
            print("Generating text embeddings...")
            text_embeddings = self.text_embedder.embed(text_data)

            print("Generating table embeddings...")
            table_embeddings = self.table_embedder.embed(table_data)

            print("Generating image embeddings...")
            image_embeddings = self.image_embedder.embed(image_data)

            # Store in vector database
            if text_embeddings:
                print("Storing text embeddings...")
                self.vector_store.insert('text',
                                      [emb['embedding'] for emb in text_embeddings],
                                      [emb['metadata'] for emb in text_embeddings])

            if table_embeddings:
                print("Storing table embeddings...")
                self.vector_store.insert('table',
                                      [emb['embedding'] for emb in table_embeddings],
                                      [emb['metadata'] for emb in table_embeddings])

            if image_embeddings:
                print("Storing image embeddings...")
                self.vector_store.insert('image',
                                      [emb['embedding'] for emb in image_embeddings],
                                      [emb['metadata'] for emb in image_embeddings])

            print("Document processing complete")
            return True
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return False

    def query(self, query_text, modality='all', top_k=5):
        """Query the system for relevant information."""
        try:
            print(f"Processing query: '{query_text}'")

            # Generate query embedding
            query_embedding = self.text_embedder.embed([{'text': query_text}])[0]['embedding']

            results = {}
            if modality in ['all', 'text']:
                print("Searching text collection...")
                results['text'] = self.vector_store.search('text', query_embedding, top_k)

            if modality in ['all', 'table']:
                print("Searching table collection...")
                results['table'] = self.vector_store.search('table', query_embedding, top_k)

            if modality in ['all', 'image']:
                print("Searching image collection...")
                results['image'] = self.vector_store.search('image', query_embedding, top_k)

            print("Query processing complete")
            return results
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {}
""",

        # API
        "src/api/__init__.py": "",

        # Endpoints
        "src/api/endpoints.py": """from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

app = FastAPI(title="Multimodal RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system = None

@app.post("/process_document")
async def process_document(file: UploadFile = File(...)):
    """Process a PDF document and store its embeddings."""
    try:
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)

        # Save uploaded file
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process document
        success = rag_system.process_document(file_path)

        if success:
            return {"message": "Document processed successfully", "filename": file.filename}
        else:
            raise HTTPException(status_code=500, detail="Error processing document")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query(query_text: str = Form(...), modality: str = Form("all"), top_k: int = Form(5)):
    """Query the system for relevant information."""
    try:
        results = rag_system.query(query_text, modality, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}
""",

        # Main application
        "main.py": """import yaml
import os
import uvicorn
from src.rag.retriever import MultimodalRAG
from src.api.endpoints import app, rag_system

def load_config():
    """Load configuration from YAML file."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_directories():
    """Create necessary directories."""
    os.makedirs('temp', exist_ok=True)
    os.makedirs('temp/vectors', exist_ok=True)

def main():
    """Main entry point for the application."""
    # Load configuration
    config = load_config()

    # Setup directories
    setup_directories()

    # Initialize RAG system
    global rag_system
    from src.api.endpoints import rag_system
    rag_system = MultimodalRAG(config)

    # Update the global variable in the API module
    import src.api.endpoints
    src.api.endpoints.rag_system = rag_system

    # Start API server
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        log_level="info"
    )

if __name__ == "__main__":
    main()
""",

        # README
        "README.md": """# Multimodal RAG System

This is a Retrieval-Augmented Generation (RAG) system that processes PDFs containing text, tables, and images.