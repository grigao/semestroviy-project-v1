import asyncio
import sys
import logging
from pathlib import Path

# Настройка путей
sys.path.insert(0, str(Path(__file__).parent))

from modules.qdrant.collections import init_collections
from modules.qdrant.search import QdrantSearch
from modules.embedding.providers.lmstudio import LMStudioEmbeddingProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("search_cli")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python search_cli.py 'query text' [limit=5]")
        sys.exit(1)

    query = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    init_collections()

    provider = LMStudioEmbeddingProvider()
    vector = await provider.embed(query)
    await provider.close()

    if not vector:
        logger.error("Failed to get embedding for query")
        sys.exit(1)

    searcher = QdrantSearch()
    results = searcher.search(vector, limit=limit)

    print(f"\n> Results for: '{query}'\n")
    for i, res in enumerate(results, 1):
        payload = res["payload"]
        print(f"{i}. Score: {res['score']:.4f}")
        print(f"   Hash: {payload.get('block_hash', 'N/A')[:16]}...")
        if payload.get("message_id"):
            print(f"   Message ID: {payload['message_id']}")
        print(f"   Source: {payload.get('source', '-')}")
        if payload.get("category"):
            print(f"   Category: {payload['category']}")
        if payload.get("keywords"):
            print(f"   Keywords: {', '.join(payload['keywords'])}")
        if payload.get("published_at"):
            print(f"   Published: {payload['published_at']}")
        if payload.get("token_count"):
            print(f"   Tokens: {payload['token_count']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())