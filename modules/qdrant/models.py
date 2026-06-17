from pydantic import BaseModel
from typing import List, Optional

class BlockPayload(BaseModel):
    block_hash: str
    message_id: str
    source: str
    channel: str
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    published_at: Optional[float] = None
    token_count: Optional[int] = None
    embedding_model: str
    embedding_version: str
    embedding_status: str
    error_message: Optional[str] = None

class QdrantPoint(BaseModel):
    """Точка для вставки в Qdrant"""
    id: str  # block_hash
    vector: List[float]
    payload: dict