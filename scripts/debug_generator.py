"""Отладка LLM-генератора дайджеста"""
import sys
import logging
from pathlib import Path
import signal
import sys

# Моментальное прерывание
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)

from modules.digest_builder.builder import builder
from modules.digest_builder.contracts import DigestRequest

# 1. Сборка
print("=" * 60)
print("ШАГ 1: Сборка дайджеста")
print("=" * 60)

# req = DigestRequest("погода в москве", max_items=5)
# req = DigestRequest(categories=["technology"], period="7d", max_items=5)
req = DigestRequest(period="7d", max_items=5)
# req = DigestRequest(categories=["technology", "science"], period="48h", max_items=5)

response = builder.build(req)

print(f"Режим: {response.mode}")
print(f"Найдено: {len(response.items)} элементов\n")
for i, item in enumerate(response.items, 1):
    print(f"{i}. [{item.source}] {item.text[:80]}...")
    print(f"   Score: {item.score:.4f} | Link: {item.post_link}")
    print()

# 2. Генерация
print("=" * 60)
print("ШАГ 2: Генерация через LLM")
print("=" * 60)

from modules.digest_builder.generator import DigestGenerator
generator = DigestGenerator()
llm_json = generator.generate(response)
print("LLM JSON:")
print(llm_json)
print()

# 3. Форматирование
print("=" * 60)
print("ШАГ 3: Форматирование в Markdown")
print("=" * 60)
from modules.digest_builder.formatter import format_digest_markdown
markdown = format_digest_markdown(llm_json, response.items)
print(markdown)