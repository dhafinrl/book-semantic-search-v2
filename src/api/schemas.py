from pydantic import BaseModel
from typing import List, Optional

class BookResult(BaseModel):
    id: Optional[int] = None
    title: str
    author: str
    genre: str
    synopsis: str
    score: float

class SearchResponse(BaseModel):
    query: str
    latency_ms: float
    results: List[BookResult]
