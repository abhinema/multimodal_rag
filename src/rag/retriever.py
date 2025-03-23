from ..extractors.text_extractor import TextExtractor
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