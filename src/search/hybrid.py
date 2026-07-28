import os
import pandas as pd
from typing import Any, Dict, List
from src.config import settings
from src.search.bm25_search import BM25Searcher
from src.search.vector_search import VectorSearcher
from src.search.reranker import CrossEncoderReranker

class HybridSearchEngine:
    def __init__(self):
        print("Initializing Hybrid Search Engine...")
        self.bm25_searcher = BM25Searcher()
        self.vector_searcher = VectorSearcher()
        self.reranker = CrossEncoderReranker()
        
        # Load metadata
        metadata_path = os.path.join(settings.INDEX_DIR, settings.METADATA_STORE_NAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        self.metadata_df = pd.read_pickle(metadata_path)
        print("Hybrid Search Engine ready.")

    def reciprocal_rank_fusion(self, list_1: List[int], list_2: List[int], k=60) -> Dict[int, float]:
        """
        Combines two ranked lists using Reciprocal Rank Fusion (RRF).
        list_1 and list_2 are lists of document indices sorted by relevance.
        k is a smoothing constant.
        Returns a dictionary mapping document index to its RRF score.
        """
        rrf_scores = {}
        
        for rank, doc_id in enumerate(list_1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        for rank, doc_id in enumerate(list_2):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        return rrf_scores

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. Lexical Search (BM25)
        bm25_results = self.bm25_searcher.search(query, top_k=settings.TOP_K_INITIAL_RETRIEVAL)
        bm25_indices = [idx for idx, _ in bm25_results]
        
        # 2. Semantic Search (FAISS)
        vector_results = self.vector_searcher.search(query, top_k=settings.TOP_K_INITIAL_RETRIEVAL)
        vector_indices = [idx for idx, _ in vector_results]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores_dict = self.reciprocal_rank_fusion(bm25_indices, vector_indices)
        
        # Sort by RRF score descending and take top N for reranking
        sorted_rrf = sorted(rrf_scores_dict.items(), key=lambda item: item[1], reverse=True)
        top_candidates = sorted_rrf[:settings.TOP_K_INITIAL_RETRIEVAL]
        candidate_indices = [idx for idx, score in top_candidates]
        
        # If no results, return empty
        if not candidate_indices:
            return []
            
        # 4. Fetch document text for reranking
        candidate_docs = []
        for idx in candidate_indices:
            row = self.metadata_df.iloc[idx]
            # Use the rich text for reranking context
            candidate_docs.append(row['rich_text'])
            
        # 5. Rerank using CrossEncoder
        rerank_scores = self.reranker.rerank(query, candidate_docs)
        
        # Combine indices with their new rerank scores
        reranked_results = list(zip(candidate_indices, rerank_scores))
        # Sort by rerank score descending
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 6. Format final output (take final top_k)
        final_results = []
        for idx, score in reranked_results[:top_k]:
            row = self.metadata_df.iloc[idx]
            result_dict = row.to_dict()
            result_dict['score'] = score
            # Remove rich_text from response if it's too large, but for now we can keep or drop it
            # result_dict.pop('rich_text', None) 
            final_results.append(result_dict)
            
        return final_results
