import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import settings

class VectorSearcher:
    def __init__(self):
        faiss_path = os.path.join(settings.INDEX_DIR, settings.FAISS_INDEX_NAME)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}. Please run pipeline first.")
        
        self.faiss_index = faiss.read_index(faiss_path)
        # Load the embedding model used for indexing
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Searches the FAISS index and returns a list of (doc_index, distance)
        Note: FAISS L2 distance is smaller = better. 
        We convert it to a similarity score (e.g. 1 / (1 + distance)) for easier integration.
        """
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx != -1:
                # Convert L2 distance to a similarity score
                similarity = 1.0 / (1.0 + float(dist))
                results.append((int(idx), similarity))
                
        return results
