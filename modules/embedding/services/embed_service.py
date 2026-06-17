import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from config import settings
from modules.embedding.schemas import BlockEmbeddingRequest, BlockEmbeddingResult, BatchEmbeddingResult
from modules.embedding.providers.lmstudio import LMStudioEmbeddingProvider

logger = logging.getLogger("embedding.service")


class EmbeddingService:
    """Сервис для генерации эмбеддингов блоков контента"""
    
    def __init__(self, provider: Optional[LMStudioEmbeddingProvider] = None):
        self.provider = provider or LMStudioEmbeddingProvider()
        self.batch_size = settings.embedding_batch_size
        self.max_retries = 3
        self.retry_delay = 0.5
    
    async def _embed_with_retry(self, text: str) -> List[float]:
        """Вызов provider.embed() с повторными попытками"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await self.provider.embed(text)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "retry attempt=%d/%d error=%s delay=%.2fs",
                        attempt + 1, self.max_retries, str(e), delay
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("retry.exhausted error=%s", str(e))
        raise last_error
    
    async def generate_embedding(self, request: BlockEmbeddingRequest) -> BlockEmbeddingResult:
        """Генерация эмбеддинга для одного блока"""
        start_time = time.time()
        
        # Валидация
        if not request.block_hash:
            return BlockEmbeddingResult.error_result(
                block_hash="unknown",
                error_message="Missing block_hash",
                source=request.source,
                model=settings.embedding_model
            )
        
        if not request.text or not request.text.strip():
            return BlockEmbeddingResult.error_result(
                block_hash=request.block_hash,
                error_message="Empty text content",
                source=request.source,
                model=settings.embedding_model
            )
        
        try:
            vector = await self._embed_with_retry(request.text)
            
            logger.debug(
                "block.embedded hash=%s source=%s dim=%d time=%.2fms",
                request.block_hash[:8], request.source, len(vector), (time.time() - start_time) * 1000
            )
            
            return BlockEmbeddingResult.success_result(
                block_hash=request.block_hash,
                vector=vector,
                source=request.source,
                model=settings.embedding_model,
                version="v1"
            )
            
        except Exception as e:
            logger.error("block.failed hash=%s source=%s error=%s", 
                        request.block_hash[:8], request.source, str(e))
            
            return BlockEmbeddingResult.error_result(
                block_hash=request.block_hash,
                error_message=str(e),
                source=request.source,
                model=settings.embedding_model
            )
    
    async def generate_embeddings_batch(self, blocks: List[Dict[str, Any]]) -> BatchEmbeddingResult:
        """Генерация эмбеддингов для нескольких блоков с использованием batch API"""
        start_time = time.time()
        
        if not blocks:
            return BatchEmbeddingResult(
                results=[],
                total_processed=0,
                total_success=0,
                total_failed=0,
                processing_time_seconds=0
            )
        
        # Преобразуем в запросы с source
        requests = [
            BlockEmbeddingRequest(
                block_hash=block.get("block_hash", ""),
                text=block.get("text", ""),
                source=block.get("source", "unknown")
            )
            for block in blocks
        ]
        
        # Фильтруем валидные запросы
        valid_indices = [i for i, req in enumerate(requests) if req.text and req.text.strip()]
        valid_requests = [requests[i] for i in valid_indices]
        
        results = [None] * len(requests)
        
        if valid_requests:
            try:
                # Batch запрос
                texts = [req.text[:10000] for req in valid_requests]
                vectors = await self.provider.embed_batch(texts)
                
                # Формируем успешные результаты
                for idx, (req, vector) in enumerate(zip(valid_requests, vectors)):
                    if vector and len(vector) > 0:
                        results[valid_indices[idx]] = BlockEmbeddingResult.success_result(
                            block_hash=req.block_hash,
                            vector=vector,
                            source=req.source,
                            model=settings.embedding_model,
                            version="v1"
                        )
                    else:
                        results[valid_indices[idx]] = BlockEmbeddingResult.error_result(
                            block_hash=req.block_hash,
                            error_message="Empty vector in batch response",
                            source=req.source,
                            model=settings.embedding_model
                        )
            except Exception as e:
                logger.error("batch.failed error=%s falling back to single", str(e))
                # Fallback на поштучную обработку
                for idx in valid_indices:
                    results[idx] = await self.generate_embedding(requests[idx])
        
        # Заполняем пропущенные (невалидные) запросы ошибками
        for i, req in enumerate(requests):
            if results[i] is None:
                results[i] = BlockEmbeddingResult.error_result(
                    block_hash=req.block_hash,
                    error_message="Invalid or empty text",
                    source=req.source,
                    model=settings.embedding_model
                )
        
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        
        processing_time = time.time() - start_time
        
        logger.info(
            "batch.complete blocks=%d success=%d failed=%d time=%.2fs",
            len(results), success_count, failed_count, processing_time
        )
        
        return BatchEmbeddingResult(
            results=results,
            total_processed=len(results),
            total_success=success_count,
            total_failed=failed_count,
            processing_time_seconds=processing_time
        )
    
    async def close(self):
        await self.provider.close()