import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_dt(v):
    """Парсит дату из строки или возвращает datetime"""
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(v)


def archive_path(root: Path, channel: str) -> Path:
    return root.resolve().parent / "output" / "raw" / f"{channel.lstrip('@')}.json"


def load_archive(path: Path) -> Dict[str, Any]:
    """Загружает архив сообщений"""
    if not path.exists():
        return {"metadata": {}, "messages": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_archive(path: Path, data: Dict[str, Any]) -> None:
    """Сохраняет архив сообщений"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_by_id(existing: List[Dict], incoming: List[Dict]):
    """Мерджит сообщения по ID, возвращает (объединённый список, добавлено, дублей)"""
    existing_ids = {m["id"] for m in existing if "id" in m}
    
    added = 0
    duplicates = 0
    merged_new = []
    
    for m in incoming:
        if m["id"] in existing_ids:
            duplicates += 1
            continue
        existing_ids.add(m["id"])
        merged_new.append(m)
        added += 1
    
    return existing + merged_new, added, duplicates


def get_coverage(meta: Dict[str, Any]):
    """Возвращает (coverage_start, coverage_end) из метаданных"""
    if not meta:
        return None, None
    
    return (
        parse_dt(meta.get("coverage_start")) if meta.get("coverage_start") else None,
        parse_dt(meta.get("coverage_end")) if meta.get("coverage_end") else None,
    )


def update_coverage(meta: Dict[str, Any], messages: List[Dict]) -> Dict[str, Any]:
    """Обновляет покрытие в метаданных на основе сообщений"""
    if not messages:
        return meta
    
    dates = [parse_dt(m["date"]) for m in messages if m.get("date")]
    if not dates:
        return meta
    
    new_start = min(dates)
    new_end = max(dates)
    
    old_start, old_end = get_coverage(meta)
    
    if old_start:
        new_start = min(old_start, new_start)
    if old_end:
        new_end = max(old_end, new_end)
    
    return {
        **meta,
        "coverage_start": new_start.isoformat(),
        "coverage_end": new_end.isoformat(),
        "updated_at": datetime.now().replace(microsecond=0).isoformat(),
    }