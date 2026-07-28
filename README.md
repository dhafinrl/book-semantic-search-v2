# Book Semantic Search Engine V2

A production-grade, high-performance semantic search engine for books. This project upgrades a basic Flask+FAISS prototype into a robust architecture utilizing **Two-Stage Retrieval** with a clean, modular design.

## 🚀 Architecture & Tech Stack

This engine uses **Hybrid Search (Lexical BM25 + Dense Vector Search via FAISS)** combined with **Reciprocal Rank Fusion (RRF)** and **Cross-Encoder Reranking** for maximum search accuracy.

- **Backend API:** FastAPI (Asynchronous, Type-Safe)
- **Vector Index:** FAISS (`faiss-cpu`)
- **Lexical Index:** BM25 (`rank_bm25`)
- **Embedding Model:** `BAAI/bge-m3` (Supports 8192 context window)
- **Reranker Model:** `BAAI/bge-reranker-base` (Cross-Encoder)
- **Caching:** Redis
- **Frontend UI:** Streamlit
- **Containerization:** Docker & Docker Compose

## 📁 Project Structure

```text
book-search-v2/
├── data/
│   ├── raw/                  # Raw dataset (CSV)
│   └── processed/            # Pickled DataFrames, BM25, and FAISS indices
├── src/
│   ├── config.py             # Environment & model configurations
│   ├── pipeline/             # Data ingestion and index building
│   ├── search/               # Core Search Engine classes (BM25, Vector, RRF, Hybrid)
│   └── api/                  # FastAPI routes and Pydantic schemas
├── frontend/
│   └── app_streamlit.py      # Streamlit search UI
├── app.py                    # FastAPI entrypoint
├── docker-compose.yml        
├── Dockerfile
└── requirements.txt
```

## 🛠️ How to Run Locally (Without Docker)

1. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Data Pipeline (Build Indices):**
   *(Note: The first run will download the `bge-m3` model weights).*
   ```bash
   python -m src.pipeline.preprocess
   python -m src.pipeline.build_index
   ```

3. **Start the API Server:**
   ```bash
   uvicorn app:app --reload
   ```
   Access the Swagger UI at: `http://localhost:8000/docs`

4. **Start the Streamlit Frontend (In a new terminal):**
   ```bash
   streamlit run frontend/app_streamlit.py
   ```
   Access the UI at: `http://localhost:8501`

## 🐳 How to Run with Docker Compose (Production Ready)

If you have Docker Desktop installed, you can spin up the API, Frontend, and Redis Cache with a single command:

```bash
docker-compose up --build
```

- **Frontend UI**: http://localhost:8501
- **API Server**: http://localhost:8000
