from .base_embedder import BaseEmbedder
import numpy as np

class TextEmbedder(BaseEmbedder):
    def __init__(self, config):
        self.config = config
        self.dimension = config['dimension']
        self.model_name = config['model']
        
        # In a production implementation, you would load a real model like:
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(self.model_name)
        
        # For demonstration, we'll use random embeddings
        print(f"Initialized TextEmbedder with model: {self.model_name}")
    
    def embed(self, text_data):
        """
        Generate embeddings for text data.
        """
        try:
            embeddings = []
            
            for item in text_data:
                # In a real implementation, you would do:
                # embedding = self.model.encode(item['text'])
                
                # For demonstration, we'll use random embeddings
                embedding = np.random.randn(self.dimension).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                
                embeddings.append({
                    'embedding': embedding,
                    'metadata': item
                })
            
            return embeddings
        except Exception as e:
            print(f"Error embedding text: {str(e)}")
            return []