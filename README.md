# Book Semantic Search Engine V2

A hybrid semantic search engine for books. This project upgrades a basic Flask+FAISS prototype into a more robust architecture utilizing **Two-Stage Retrieval** with a clean, modular design, complete with an Admin Dashboard for data management.

## 🚀 Architecture & Tech Stack

This engine uses **Hybrid Search (Lexical BM25 + Dense Vector Search via FAISS)** combined with **Reciprocal Rank Fusion (RRF)** and **Cross-Encoder Reranking** for maximum search accuracy.

- **Backend API:** FastAPI (Asynchronous, Type-Safe)
- **Database:** SQLite (SQLAlchemy ORM) for dynamic book management
- **Vector Index:** FAISS (`faiss-cpu`)
- **Lexical Index:** BM25 (`rank_bm25`)
- **Embedding Model:** `BAAI/bge-m3` (Supports 8192 context window)
- **Reranker Model:** `BAAI/bge-reranker-base` (Cross-Encoder)
- **Caching:** Redis
- **Frontend UI:** Streamlit (Features Search Engine & Admin Dashboard)
- **Containerization:** Docker & Docker Compose

## 📁 Project Structure

```text
book-search-v2/
├── data/                     # Data and DB storage (auto-generated)
├── src/
│   ├── config.py             # Environment & model configurations
│   ├── db/                   # Database models and session management
│   ├── pipeline/             # Data ingestion and AI index building
│   ├── search/               # Core Search Engine classes (BM25, Vector, RRF, Hybrid)
│   └── api/                  # FastAPI routes and Pydantic schemas
├── frontend/
│   └── app_streamlit.py      # Streamlit search UI and Admin Dashboard
├── app.py                    # FastAPI entrypoint (Auto-initializes DB & Indices)
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

2. **Start the API Server:**
   *(Note: The first time you run this, it will automatically seed the SQLite database from CSV and download the AI models to build the FAISS/BM25 indices. This may take a few minutes).*
   ```bash
   uvicorn app:app --reload
   ```
   Access the Swagger UI at: `http://localhost:8000/docs`

3. **Start the Streamlit Frontend (In a new terminal):**
   ```bash
   streamlit run frontend/app_streamlit.py
   ```
   Access the UI at: `http://localhost:8501`

## 🐳 How to Run with Docker Compose (Production Ready)

If you have Docker Desktop installed, you can spin up the API, Frontend, and Redis Cache with a single command:

```bash
docker-compose up --build
```

*(Note: The HuggingFace models are cached in a named volume so they will only be downloaded on the very first run, saving significant time on subsequent builds).*

- **Frontend UI**: http://localhost:8501
- **API Server**: http://localhost:8000
