"""Статистика категорий в enriched-файлах"""
import json
from collections import Counter
from pathlib import Path

ENRICHED_DIR = Path(__file__).parent.parent / "modules" / "output" / "enriched"

cats = Counter()

for f in ENRICHED_DIR.glob("*.json"):
    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for msg in data.get("messages", []):
        c = msg.get("enrichment", {}).get("category")
        if c:
            cats[c] += 1

print("Категории в enriched:")
print("-" * 30)
for cat, count in cats.most_common():
    print(f"{cat:20s} {count}")
print("-" * 30)
print(f"Всего: {sum(cats.values())}")