from sentence_transformers import SentenceTransformer
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