import numpy as np
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