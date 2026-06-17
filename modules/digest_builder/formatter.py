"""Форматирование дайджеста в Markdown"""
import json
from typing import List
from modules.digest_builder.contracts import DigestItem

def format_digest_markdown(llm_json: str, items: List[DigestItem]) -> str:
    try:
        data = json.loads(llm_json)
    except json.JSONDecodeError:
        data = {"title": "Дайджест", "items": []}
    
    lines = [f"# {data.get('title', 'Дайджест')}", ""]
    
    for i, entry in enumerate(data.get("items", []), 1):
        summary = entry.get("summary", "")
        source_indices = entry.get("source_indices", [])
        
        if not source_indices:
            lines.append(f"{i}. {summary}")
            lines.append("")
            continue
        
        # Берём ПЕРВЫЙ индекс из группы LLM
        first_idx = source_indices[0] - 1  # индексы в LLM с 1
        
        if 0 <= first_idx < len(items):
            item = items[first_idx]
            source = item.source
            link = item.post_link
        else:
            source = "?"
            link = None
        
        if link:
            lines.append(f"{i}. {summary} — [{source}]({link})")
        else:
            lines.append(f"{i}. {summary} — {source}")
        lines.append("")
    
    return "\n".join(lines)