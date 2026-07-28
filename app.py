from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes import router as api_router
from src.search.hybrid import HybridSearchEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting up the FastAPI server...")
    # Initialize the heavy machine learning models once during startup
    app.state.search_engine = HybridSearchEngine()
    print("Machine learning models loaded into memory.")
    yield
    # Shutdown logic
    print("Shutting down the server...")
    app.state.search_engine = None

app = FastAPI(
    title="Book Semantic Search Engine API",
    description="High-performance hybrid search engine using BM25 and FAISS with Cross-Encoder reranking.",
    version="2.0.0",
    lifespan=lifespan
)

# Include the API routes
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    # When running directly with python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
