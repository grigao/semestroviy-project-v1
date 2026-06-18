"""Форматирование дайджеста в Markdown"""
import json
from typing import List
from modules.digest_builder.contracts import DigestItem


def format_digest_markdown(llm_json: str, items: List[DigestItem], mode: str = "search") -> str:
    try:
        data = json.loads(llm_json)
    except json.JSONDecodeError:
        data = {"title": "Дайджест", "items": []}
    
    lines = [f"# {data.get('title', 'Дайджест')}", ""]
    
    llm_items = data.get("items", [])
    item_number = 1  # свой счётчик, без пропусков
    
    for entry in llm_items:
        summary = entry.get("summary", "")
        source_indices = entry.get("source_indices", [])
        
        if not source_indices:
            continue
        
        first_idx = source_indices[0] - 1
        if 0 <= first_idx < len(items):
            item = items[first_idx]
            source = item.source
            link = item.post_link
        else:
            source = "?"
            link = None
        
        if link:
            lines.append(f"{item_number}. {summary} — [{source}]({link})")
        else:
            lines.append(f"{item_number}. {summary} — {source}")
        lines.append("")
        item_number += 1
    
    # Footer только для sub режима
    if mode == "subscription":
        total_input = len(items)
        total_output = len(llm_items)
        if total_output < total_input:
            lines.append("---")
            lines.append(f"*Показано {total_output} из {total_input} новостей.*")
    
    return "\n".join(lines)