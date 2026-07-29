from sentence_transformers import CrossEncoder
from src.config import settings

class CrossEncoderReranker:
    def __init__(self):
        # Use CrossEncoder for reranking
        self.model = CrossEncoder(settings.RERANKER_MODEL_NAME)
        
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Reranks a list of documents based on a query.
        Returns a list of scores corresponding to each document.
        """
        if not documents:
            return []
            
        # CrossEncoder expects a list of (query, document) pairs
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Convert to list of python floats
        return [float(score) for score in scores]
