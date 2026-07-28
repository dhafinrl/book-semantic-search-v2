import os
import pickle
import numpy as np
from src.config import settings
from src.pipeline.build_index import tokenize_corpus

class BM25Searcher:
    def __init__(self):
        bm25_path = os.path.join(settings.INDEX_DIR, settings.BM25_INDEX_NAME)
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Please run pipeline first.")
        
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
            
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Searches the BM25 index and returns a list of (doc_index, score)
        """
        tokenized_query = tokenize_corpus([query])[0]
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices sorted by score descending
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            score = float(scores[idx])
            if score > 0:  # Only include if there's some matching
                results.append((int(idx), score))
                
        return results
