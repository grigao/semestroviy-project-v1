"""Параметры digest builder по умолчанию"""
from dataclasses import dataclass

@dataclass
class DigestConfig:
    # Дедупликация
    t_base: float = 0.92        # базовый порог косинуса
    delta: float = 0.15          # максимальное снижение порога
    sigma_t: float = 6.0       # часы, после которых тема может повториться
    
    # MMR
    lambda_mmr: float = 0.7     # релевантность vs diversity
    
    # Лимиты
    max_candidates: int = 50   # первичный отбор из Qdrant
    max_items: int = 10         # финальный дайджест
    
    # Модели
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

# Экземпляр по умолчанию
config = DigestConfig()