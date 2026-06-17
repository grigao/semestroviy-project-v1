from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from .client import client
from config import settings

def init_collections():
    try:
        client.get_collection(settings.qdrant_collection)
        return False  # уже существует
    except (UnexpectedResponse, Exception):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE
            )
        )
        return True