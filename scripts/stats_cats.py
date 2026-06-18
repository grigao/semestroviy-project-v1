"""Статистика категорий в enriched-файлах"""
import json
import sys
from collections import Counter
from pathlib import Path

ENRICHED_DIR = Path(__file__).parent.parent / "modules" / "output" / "enriched"

# Если указан источник — фильтруем
source_filter = sys.argv[1] if len(sys.argv) > 1 else None

cats = Counter()

for f in ENRICHED_DIR.glob("*.json"):
    if source_filter and f.stem != source_filter:
        continue
    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for msg in data.get("messages", []):
        c = msg.get("enrichment", {}).get("category")
        if c:
            cats[c] += 1

label = f"для {source_filter}" if source_filter else "по всем каналам"
print(f"Категории {label}:")
print("-" * 30)
for cat, count in cats.most_common():
    print(f"{cat:20s} {count}")
print("-" * 30)
print(f"Всего: {sum(cats.values())}")