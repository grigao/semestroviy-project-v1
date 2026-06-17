"""Единая точка доступа к Qdrant"""
from qdrant_client import QdrantClient
from config import settings

client = QdrantClient(url=settings.qdrant_url)