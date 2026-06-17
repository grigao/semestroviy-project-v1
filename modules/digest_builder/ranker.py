"""Ранжирование: precision mode (реранкер) и diversity mode (MMR)"""
import logging
from typing import List, Dict, Tuple
import numpy as np

from modules.digest_builder.config import config

logger = logging.getLogger("digest.ranker")


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Косинусное сходство между двумя векторами"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def build_similarity_matrix(vectors: List[List[float]]) -> List[List[float]]:
    """Строит матрицу косинусных сходств между всеми векторами"""
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def precision_rank(
    candidates: List[Dict],
    reranker_scores: List[float],
    vectors: List[List[float]],
    max_items: int = None
) -> Tuple[List[int], List[float]]:
    """
    Precision mode: ранжирование по реранкер-скору + дедупликация.
    
    Returns:
        (индексы отобранных, веса для дедупликации)
    """
    max_items = max_items or config.max_items
    n = len(candidates)
    
    if n == 0:
        return [], []
    
    # Строим матрицу сходств для дедупликации
    sim_matrix = build_similarity_matrix(vectors)
    
    # Веса = скоры реранкера
    weights = reranker_scores.copy()
    
    # Дедупликация
    from modules.digest_builder.deduplicator import deduplicate
    selected_indices = deduplicate(candidates, sim_matrix, weights)
    
    # Сортируем отобранные по весу и берём топ
    selected_with_weights = [(i, weights[i]) for i in selected_indices]
    selected_with_weights.sort(key=lambda x: x[1], reverse=True)
    
    final_indices = [i for i, _ in selected_with_weights[:max_items]]
    final_weights = [weights[i] for i in final_indices]
    
    return final_indices, final_weights


def mmr_rank(
    candidates: List[Dict],
    query_vector: List[float],
    reranker_scores: List[float],
    vectors: List[List[float]],
    max_items: int = None,
    lambda_mmr: float = None
) -> Tuple[List[int], List[float]]:
    """
    Diversity mode (MMR): баланс релевантности и разнообразия.
    
    MMR(d_i) = λ * Sim1(d_i, q) - (1-λ) * max_{d_j ∈ S} Sim2(d_i, d_j)
    
    Sim1 — реранкер-скор (релевантность запросу)
    Sim2 — косинусное сходство между документами
    
    Returns:
        (индексы отобранных, MMR-скоры)
    """
    max_items = max_items or config.max_items
    lambda_mmr = lambda_mmr or config.lambda_mmr
    n = len(candidates)
    
    if n == 0:
        return [], []
    
    # Нормализуем реранкер-скоры в [0, 1]
    r_scores = np.array(reranker_scores)
    if r_scores.max() > r_scores.min():
        r_scores = (r_scores - r_scores.min()) / (r_scores.max() - r_scores.min())
    else:
        r_scores = np.ones_like(r_scores) * 0.5
    
    # Строим матрицу сходств
    sim_matrix = build_similarity_matrix(vectors)
    
    selected = []
    remaining = set(range(n))
    
    while remaining and len(selected) < max_items:
        mmr_scores = {}
        
        for i in remaining:
            # Релевантность запросу
            relevance = r_scores[i]
            
            # Максимальное сходство с уже отобранными
            if selected:
                max_sim = max(sim_matrix[i][j] for j in selected)
            else:
                max_sim = 0.0
            
            mmr = lambda_mmr * relevance - (1 - lambda_mmr) * max_sim
            mmr_scores[i] = mmr
        
        # Выбираем максимум
        best = max(mmr_scores, key=mmr_scores.get)
        selected.append(best)
        remaining.remove(best)
    
    final_weights = [mmr_scores.get(i, 0.0) for i in selected]
    
    return selected, final_weights


def rank_candidates(
    candidates: List[Dict],
    query_vector: List[float],
    reranker_scores: List[float],
    vectors: List[List[float]],
    mode: str = "search",
    max_items: int = None
) -> Tuple[List[int], List[float]]:
    """
    Единая точка входа для ранжирования.
    
    Args:
        candidates: список кандидатов (словари с полями блоков)
        query_vector: вектор запроса (может быть None для subscription)
        reranker_scores: скоры реранкера для каждого кандидата
        vectors: векторы кандидатов
        mode: "search" (precision) или "subscription" (diversity)
        max_items: максимум элементов в результате
    
    Returns:
        (индексы отобранных кандидатов, итоговые скоры)
    """
    if mode == "search":
        logger.info("Ranking mode: precision")
        return precision_rank(candidates, reranker_scores, vectors, max_items)
    else:
        logger.info("Ranking mode: diversity (MMR)")
        return mmr_rank(
            candidates, query_vector, reranker_scores, vectors, max_items
        )