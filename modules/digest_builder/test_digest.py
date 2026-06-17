"""Промежуточный тест модулей digest_builder"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=== Тест 0: Импорты ===")

# 1. Конфиг и контракты
print("\n1. config + contracts...")
from modules.digest_builder.config import config
from modules.digest_builder.contracts import DigestRequest, DigestItem, DigestResponse
print(f"   OK: t_base={config.t_base}, max_items={config.max_items}")

# 2. Фильтры
print("\n2. filters...")
from modules.digest_builder.filters import build_filter, parse_period
req = DigestRequest(categories=["technology"], sources=["habr_com"], period="24h")
f = build_filter(req)
print(f"   OK: filter={f}")

# 3. Рера́нкер (ленивая загрузка — модель скачается при первом вызове)
print("\n3. reranker (loading model, may download ~1.1GB)...")
from modules.digest_builder.reranker import Reranker
reranker = Reranker()
docs = [
    "Память LLM: как работает контекстное окно в трансформерах",
    "Новый самокат Whoosh появился в Москве",
    "BPE и токенизация: разбор алгоритма сжатия словаря"
]
query = "память llm"
scores = reranker.rerank(query, docs)
print(f"   OK: query='{query}'")
for idx, score in scores:
    print(f"   [{score:.4f}] {docs[idx][:60]}...")

# 4. Дедупликация
print("\n4. deduplicator...")
from modules.digest_builder.deduplicator import compute_dynamic_threshold
t1 = "2026-06-16T10:00:00"
t2 = "2026-06-16T14:00:00"
th = compute_dynamic_threshold(t1, t2)
print(f"   OK: threshold(4h diff) = {th:.4f}")
t3 = "2026-06-15T10:00:00"
th2 = compute_dynamic_threshold(t1, t3)
print(f"   OK: threshold(24h diff) = {th2:.4f}")

# 5. Ранкер
print("\n5. ranker...")
from modules.digest_builder.ranker import compute_cosine_similarity, build_similarity_matrix
import numpy as np
v1 = list(np.random.randn(1024))
v2 = list(np.random.randn(1024))
sim = compute_cosine_similarity(v1, v2)
print(f"   OK: random vectors cosine={sim:.4f}")

print("\n=== Все тесты пройдены ===")