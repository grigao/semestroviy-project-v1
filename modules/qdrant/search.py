"""Семантический поиск по коллекции"""
from typing import List, Optional, Dict, Any
from qdrant_client.models import Filter
from .client import client
from config import settings

class QdrantSearch:
    def search(self, vector, limit=10, filter=None):
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=limit,
            query_filter=filter,
            with_vectors=True
        )
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload, "vector": hit.vector}
            for hit in results
        ]

searcher = QdrantSearch()