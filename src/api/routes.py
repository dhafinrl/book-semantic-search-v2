import time
import json
import hashlib
import redis
from typing import Optional
from fastapi import APIRouter, Query, Request, Depends, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from src.config import settings
from src.api.schemas import SearchResponse, BookCreate, BookDB
from src.db.database import get_db
from src.db.models import Book

router = APIRouter()

# Attempt to connect to Redis
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    # Ping to check if actually running
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis cache connected successfully.")
except redis.ConnectionError:
    print("Warning: Redis is not available. Running without cache (Good for local development).")
    REDIS_AVAILABLE = False

def get_cache_key(query: str, top_k: int) -> str:
    """Generate a unique deterministic hash for the query."""
    key_string = f"{query.lower().strip()}_{top_k}"
    return f"search_cache:{hashlib.md5(key_string.encode()).hexdigest()}"

@router.get("/search", response_model=SearchResponse)
async def search_books(
    request: Request,
    q: str = Query(..., description="The search query"),
    top_k: int = Query(settings.TOP_K_FINAL_RESULTS, ge=1, le=50, description="Number of results to return")
):
    start_time = time.time()
    
    # 1. Check Redis Cache
    if REDIS_AVAILABLE:
        cache_key = get_cache_key(q, top_k)
        cached_result = redis_client.get(cache_key)
        if cached_result:
            latency = (time.time() - start_time) * 1000
            # Cached result is already a JSON string, parse it back 
            # to match the response_model, or return Response directly.
            # Returning parsed dict is easiest for Pydantic.
            results = json.loads(cached_result)
            return SearchResponse(query=q, latency_ms=latency, results=results)
            
    # 2. Not in cache, perform actual search
    # Retrieve the search engine instance attached to the app state
    search_engine = request.app.state.search_engine
    
    # Run the heavy, synchronous search function in a background thread
    # to avoid blocking the FastAPI async event loop
    results = await run_in_threadpool(search_engine.search, q, top_k)
    
    latency = (time.time() - start_time) * 1000
    
    # 3. Save to Redis Cache
    if REDIS_AVAILABLE:
        # Convert results to JSON string
        try:
            cache_key = get_cache_key(q, top_k)
            redis_client.setex(
                name=cache_key,
                time=settings.REDIS_CACHE_EXPIRE,
                value=json.dumps(results)
            )
        except Exception as e:
            print(f"Failed to set cache: {e}")
            
    return SearchResponse(query=q, latency_ms=latency, results=results)
@router.post('/books', response_model=BookDB)
def create_book(book: BookCreate, db: Session = Depends(get_db)):
    db_book = Book(title=book.title, author=book.author, genre=book.genre, synopsis=book.synopsis)
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    # Invalidate search cache
    if REDIS_AVAILABLE:
        redis_client.flushdb()
    return db_book

@router.post('/index/sync')
async def sync_index(request: Request, background_tasks: BackgroundTasks):
    from src.pipeline.preprocess import preprocess_data
    from src.pipeline.build_index import run_indexing
    def rebuild_job():
        preprocess_data()
        run_indexing()
        # After rebuilding, refresh the search engine in memory
        request.app.state.search_engine.reload_indices()
    background_tasks.add_task(rebuild_job)
    return {'message': 'Index rebuild started in background.'}
