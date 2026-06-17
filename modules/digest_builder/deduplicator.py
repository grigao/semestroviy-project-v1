"""Дедупликация с динамическим порогом и жадным отбором"""
import math
import logging
from typing import List, Dict, Tuple
from datetime import datetime

from modules.digest_builder.config import config

logger = logging.getLogger("digest.deduplicator")


def compute_dynamic_threshold(
    time_i: str, time_j: str,
    t_base: float = None,
    delta: float = None,
    sigma_t: float = None
) -> float:
    t_base = t_base or config.t_base
    delta = delta or config.delta
    sigma_t = sigma_t or config.sigma_t
    
    try:
        ti = datetime.fromisoformat(time_i)
        tj = datetime.fromisoformat(time_j)
        diff_hours = abs((ti - tj).total_seconds()) / 3600.0
    except (ValueError, TypeError):
        return t_base
    
    # Было: t_base - delta * exp(-diff / sigma_t) — инвертировано
    # Правильно: t_base - delta * (1 - exp(-diff / sigma_t))
    # При diff=0: t_base (максимальный порог)
    # При diff→∞: t_base - delta (минимальный порог)
    return t_base - delta * (1 - math.exp(-diff_hours / sigma_t))


def build_similarity_graph(
    candidates: List[Dict],
    similarity_matrix: List[List[float]]
) -> List[Tuple[int, int]]:
    """
    Строит граф дубликатов: ребро между i и j, если сходство > порог.
    
    Args:
        candidates: список кандидатов с полями published_at
        similarity_matrix: матрица косинусных сходств между кандидатами
    
    Returns:
        Список рёбер (i, j)
    """
    edges = []
    n = len(candidates)
    
    for i in range(n):
        for j in range(i + 1, n):
            threshold = compute_dynamic_threshold(
                candidates[i].get("published_at"),
                candidates[j].get("published_at")
            )
            
            if similarity_matrix[i][j] > threshold:
                edges.append((i, j))
    
    logger.debug("Graph built: %d nodes, %d edges", n, len(edges))
    return edges


def greedy_maximal_independent_set(
    candidates: List[Dict],
    edges: List[Tuple[int, int]],
    weights: List[float]
) -> List[int]:
    """
    Жадный отбор максимального независимого множества.
    
    Сортируем вершины по убыванию веса, добавляем, если нет рёбер с уже выбранными.
    
    Returns:
        Индексы отобранных кандидатов
    """
    n = len(candidates)
    
    # Строим списки смежности
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Сортируем индексы по убыванию веса
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    selected = []
    selected_set = set()
    
    for i in sorted_indices:
        # Проверяем, нет ли рёбер с уже выбранными
        if not adj[i].intersection(selected_set):
            selected.append(i)
            selected_set.add(i)
    
    logger.debug("Greedy MIS: %d selected out of %d", len(selected), n)
    return selected


def deduplicate(
    candidates: List[Dict],
    similarity_matrix: List[List[float]],
    weights: List[float]
) -> List[int]:
    """
    Полный пайплайн дедупликации:
    1. Строит граф с динамическим порогом
    2. Жадно отбирает независимое множество
    
    Returns:
        Индексы отобранных кандидатов
    """
    edges = build_similarity_graph(candidates, similarity_matrix)
    selected = greedy_maximal_independent_set(candidates, edges, weights)
    return selected