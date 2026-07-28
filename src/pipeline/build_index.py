import os
import pickle
import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import settings

def load_processed_data() -> pd.DataFrame:
    input_path = os.path.join(settings.DATA_PROCESSED_DIR, settings.PROCESSED_DATA_FILE)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed data not found at {input_path}")
    return pd.read_pickle(input_path)

def tokenize_corpus(corpus: list[str]) -> list[list[str]]:
    """Simple whitespace tokenization for BM25."""
    return [doc.lower().split(" ") for doc in corpus]

def run_indexing():
    print("Loading processed data...")
    df = load_processed_data()
    corpus = df['rich_text'].tolist()
    
    # 1. Build BM25 Index
    print("Building BM25 index...")
    tokenized_corpus = tokenize_corpus(corpus)
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_path = os.path.join(settings.INDEX_DIR, settings.BM25_INDEX_NAME)
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {bm25_path}")
    
    # 2. Build FAISS Index
    print(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    
    print("Encoding corpus to dense vectors...")
    # Using encode with show_progress_bar
    embeddings = model.encode(corpus, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    # BGE models use cosine similarity, which is equivalent to Inner Product (if normalized) 
    # but IndexFlatL2 is also fine for standard nearest neighbors. We'll use L2 for simplicity,
    # or IndexFlatIP if normalized. Let's stick to IndexFlatL2.
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    
    faiss_path = os.path.join(settings.INDEX_DIR, settings.FAISS_INDEX_NAME)
    faiss.write_index(faiss_index, faiss_path)
    print(f"FAISS index saved to {faiss_path}")
    
    # 3. Save Metadata
    metadata_path = os.path.join(settings.INDEX_DIR, settings.METADATA_STORE_NAME)
    # We can save the dataframe to retrieve it later by index ID
    df.to_pickle(metadata_path)
    print(f"Metadata saved to {metadata_path}")
    
    print("Indexing completed successfully!")

if __name__ == "__main__":
    run_indexing()
