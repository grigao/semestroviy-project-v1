"""Отладка digest builder по шагам"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from modules.qdrant.search import QdrantSearch
from modules.embedding.providers.lmstudio import LMStudioEmbeddingProvider
from modules.digest_builder.filters import build_filter
from modules.digest_builder.contracts import DigestRequest
from modules.digest_builder.config import config

async def main():
    query = "память llm"
    print(f"1. Запрос: '{query}'")
    
    # 2. Вектор запроса
    print("\n2. Получаем вектор запроса...")
    embedder = LMStudioEmbeddingProvider()
    vector = await embedder.embed(query)
    await embedder.close()
    if vector:
        print(f"   OK: длина={len(vector)}, первые 5 значений={vector[:5]}")
    else:
        print("   ОШИБКА: пустой вектор")
        return
    
    # 3. Поиск без фильтра
    print("\n3. Поиск в Qdrant без фильтра...")
    searcher = QdrantSearch()
    results = searcher.search(vector, limit=5)
    print(f"   Найдено: {len(results)}")
    for r in results:
        print(f"   - {r['payload'].get('block_hash', '?')[:16]}... score={r['score']:.4f} source={r['payload'].get('source')}")
    
    # 4. Поиск с фильтром
    print("\n4. Поиск с фильтром...")
    request = DigestRequest(query=query)
    f = build_filter(request)
    print(f"   Фильтр: {f}")
    results_f = searcher.search(vector, limit=5, filter=f)
    print(f"   Найдено: {len(results_f)}")
    
    # 5. Проверка поля published_at
    print("\n5. Проверка формата published_at...")
    if results:
        sample = results[0]["payload"]
        pa = sample.get("published_at")
        print(f"   published_at = {pa} (тип: {type(pa).__name__})")

if __name__ == "__main__":
    asyncio.run(main())