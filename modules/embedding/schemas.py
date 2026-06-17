"""
Схемы данных для модуля embedding
Контракты между слоями
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class BlockEmbeddingRequest:
    """Запрос на генерацию эмбеддинга для блока"""
    block_hash: str
    text: str
    source: str  # источник (канал/чат), например "habr_com"


@dataclass
class BlockEmbeddingResult:
    """
    Полный результат генерации эмбеддинга для передачи в Qdrant.
    Содержит и вектор, и метаданные.
    """
    block_hash: str
    vector: List[float]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    
    @classmethod
    def success_result(cls, block_hash: str, vector: List[float], 
                       source: str, model: str, version: str = "v1") -> "BlockEmbeddingResult":
        """Фабрика успешного результата"""
        return cls(
            block_hash=block_hash,
            vector=vector,
            metadata={
                "block_hash": block_hash,
                "source": source,
                "embedding_model": model,
                "embedding_version": version,
                "embedding_dimension": len(vector),
                "embedding_status": "completed",
                "generated_at": datetime.utcnow().isoformat()
            },
            success=True
        )
    
    @classmethod
    def error_result(cls, block_hash: str, error_message: str,
                     source: str, model: str, version: str = "v1") -> "BlockEmbeddingResult":
        """Фабрика результата с ошибкой"""
        return cls(
            block_hash=block_hash,
            vector=[],
            metadata={
                "block_hash": block_hash,
                "source": source,
                "embedding_model": model,
                "embedding_version": version,
                "embedding_dimension": 0,
                "embedding_status": "failed",
                "generated_at": datetime.utcnow().isoformat(),
                "error_message": error_message
            },
            success=False,
            error_message=error_message
        )
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Только метаданные (без вектора) для сохранения на диск"""
        return self.metadata.copy()


@dataclass
class BatchEmbeddingResult:
    """Результат batch-обработки"""
    results: List[BlockEmbeddingResult]
    total_processed: int
    total_success: int
    total_failed: int
    processing_time_seconds: float