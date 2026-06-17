"""Построение Qdrant-фильтров из DigestRequest"""
from datetime import datetime, timedelta
from typing import Optional
from qdrant_client.models import Filter, FieldCondition, Range, MatchAny

from modules.digest_builder.contracts import DigestRequest


def parse_period(period: str) -> Optional[float]:
    """
    Преобразует строку периода в Unix timestamp (float).
    Qdrant сравнивает числа, а не строки дат.
    """
    now = datetime.utcnow()
    
    if period == "24h":
        dt = now - timedelta(hours=24)
    elif period == "7d":
        dt = now - timedelta(days=7)
    elif period == "48h":
        dt = now - timedelta(hours=48)
    elif period == "since_last":
        dt = now - timedelta(hours=24)
    else:
        try:
            hours = int(period.replace("h", ""))
            dt = now - timedelta(hours=hours)
        except ValueError:
            return None
    
    return dt.timestamp()

def build_filter(request: DigestRequest) -> Optional[Filter]:
    """Строит Qdrant Filter из параметров запроса"""
    conditions = []
    
    if request.categories:
        conditions.append(
            FieldCondition(
                key="category",
                match=MatchAny(any=request.categories)
            )
        )
    
    if request.sources:
        conditions.append(
            FieldCondition(
                key="source",
                match=MatchAny(any=request.sources)
            )
        )
    
    if request.period:
        since = parse_period(request.period)
        if since is not None:
            conditions.append(
                FieldCondition(
                    key="published_at",
                    range=Range(gte=since)
                )
            )
    
    return Filter(must=conditions) if conditions else None
