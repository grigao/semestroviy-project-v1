"""Загрузка точек в Qdrant"""
from typing import List, Dict, Any
from uuid import uuid5, NAMESPACE_DNS
from qdrant_client.models import PointStruct
from .client import client
from config import settings

def hash_to_uuid(block_hash: str) -> str:
    """Преобразует SHA256 хеш в UUID v5"""
    return str(uuid5(NAMESPACE_DNS, block_hash))

class QdrantUploader:
    def upsert(self, point: Dict[str, Any]):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[
                PointStruct(
                    id=hash_to_uuid(point["id"]),
                    vector=point["vector"],
                    payload=point["payload"]
                )
            ]
        )

    def upsert_batch(self, points: List[Dict[str, Any]]):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[
                PointStruct(
                    id=hash_to_uuid(p["id"]),
                    vector=p["vector"],
                    payload=p["payload"]
                )
                for p in points
            ]
        )

uploader = QdrantUploader()