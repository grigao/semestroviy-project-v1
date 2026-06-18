"""Семантический поиск по коллекции"""
from typing import List, Optional, Dict, Any
from qdrant_client.models import Filter
from .client import client
from config import settings

class QdrantSearch:
    """Поисковый движок поверх Qdrant"""
    
    def search(
        self,
        vector: List[float],
        limit: int = 10,
        filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Поиск ближайших соседей."""
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=limit,
            query_filter=filter,
            with_vectors=True,
        )
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload, "vector": hit.vector}
            for hit in results
        ]
    
    def scroll(
        self,
        filter: Optional[Filter] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получить все точки по фильтру без семантического поиска."""
        results, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=filter,
            limit=limit,
            with_vectors=True,
        )
        return [
            {"id": hit.id, "score": 1.0, "payload": hit.payload, "vector": hit.vector}
            for hit in results
        ]

searcher = QdrantSearch()