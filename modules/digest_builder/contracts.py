"""Контракты данных для digest builder"""
from pydantic import BaseModel
from typing import List, Optional

class DigestRequest(BaseModel):
    query: Optional[str] = None
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    period: Optional[str] = None
    max_items: int = 10

class DigestItem(BaseModel):
    block_hash: str
    message_id: str
    source: str
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    published_at: Optional[float] = None
    text: str
    score: float
    post_link: Optional[str] = None

class DigestResponse(BaseModel):
    query: Optional[str] = None
    mode: str
    items: List[DigestItem]
    generated_at: str