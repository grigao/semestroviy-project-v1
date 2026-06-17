"""Сброс embedding_status в enriched-файлах"""
import json
from pathlib import Path

enriched_dir = Path("modules/output/enriched")

for file_path in enriched_dir.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    changed = False
    if "messages" in data:
        for msg in data["messages"]:
            for block in msg.get("content_blocks", []):
                if block.get("embedding_status") == "completed":
                    block["embedding_status"] = "pending"
                    changed = True
    
    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Сброшены статусы в: {file_path.name}")

print("Готово")