"""Рера́нкер на основе BGE-Reranker-v2-m3 через sentence-transformers"""
import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from modules.digest_builder.config import config

logger = logging.getLogger("digest.reranker")


class Reranker:
    """Кросс-энкодер для точного ранжирования пар запрос-документ"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.reranker_model
        self._model = None
    
    @property
    def model(self) -> CrossEncoder:
        """Ленивая загрузка модели"""
        if self._model is None:
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded")
        return self._model
    
    def rerank(
        self, query: str, documents: List[str], return_scores: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Ранжирует документы по релевантности запросу.
        
        Args:
            query: поисковый запрос
            documents: список текстов документов
            return_scores: возвращать ли скоры (True) или только порядок (False)
        
        Returns:
            Список (индекс_документа, скор), отсортированный по убыванию скора
        """
        if not documents:
            return []
        
        # Формируем пары (query, doc)
        pairs = [(query, doc) for doc in documents]
        
        # Прогоняем через кросс-энкодер
        scores = self.model.predict(pairs)
        
        # Сортируем по убыванию скора
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores
    
    def rerank_top_k(
        self, query: str, documents: List[str], k: int
    ) -> List[Tuple[int, float]]:
        """Ранжирует и возвращает топ-k"""
        ranked = self.rerank(query, documents)
        return ranked[:k]