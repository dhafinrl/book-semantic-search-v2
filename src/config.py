import os
from pydantic_settings import BaseSettings

# Define base directory based on this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    # Paths
    DATA_RAW_DIR: str = os.path.join(BASE_DIR, "data", "raw")
    DATA_PROCESSED_DIR: str = os.path.join(BASE_DIR, "data", "processed")
    INDEX_DIR: str = os.path.join(BASE_DIR, "data", "processed")
    RAW_DATA_FILE: str = "dummy_books.csv"
    PROCESSED_DATA_FILE: str = "processed_books.pkl"
    DATABASE_URL: str = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'books.db')}"
    
    # Model Configurations
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    
    # Search Configurations
    FAISS_INDEX_NAME: str = "faiss_index.bin"
    BM25_INDEX_NAME: str = "bm25_index.pkl"
    METADATA_STORE_NAME: str = "metadata.pkl"
    TOP_K_INITIAL_RETRIEVAL: int = 50
    TOP_K_FINAL_RESULTS: int = 10
    
    # Redis Cache Configurations
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_CACHE_EXPIRE: int = 3600  # 1 hour

    class Config:
        env_file = ".env"

settings = Settings()
