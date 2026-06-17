"""
Провайдер для LM Studio API
Единственное место общения с LM Studio.
Никакой бизнес-логики, только HTTP вызовы.
"""

import httpx
import logging
from typing import List
from config import settings

logger = logging.getLogger("embedding.lmstudio")


class LMStudioEmbeddingProvider:
    """Провайдер эмбеддингов через LM Studio API"""
    
    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        self.base_url = (base_url or settings.lmstudio_base_url).rstrip('/')
        self.model = model or settings.embedding_model
        self.timeout = timeout or settings.lmstudio_timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
        return self._client
    
    async def embed(self, text: str) -> List[float]:
        """
        Получить эмбеддинг для одного текста.
        
        Args:
            text: Текст для эмбеддинга (не пустой)
            
        Returns:
            List[float]: Вектор эмбеддинга
            
        Raises:
            ValueError: если текст пустой
            Exception: при ошибках HTTP, таймауте, невалидном ответе
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Ограничение длины (LM Studio может иметь лимиты)
        text = text[:10000]
        
        payload = {
            "input": text,
            "model": self.model,
            "encoding_format": "float"
        }
        
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data or not data["data"]:
                raise Exception("Empty or invalid response: no 'data' field")
            
            embedding = data["data"][0].get("embedding")
            if embedding is None:
                raise Exception("No embedding vector in response")
            
            # Валидация размерности (логируем, но не блокируем)
            if len(embedding) != settings.embedding_dimension:
                logger.warning(
                    "dimension.mismatch expected=%d got=%d",
                    settings.embedding_dimension, len(embedding)
                )
            
            return embedding
            
        except httpx.TimeoutException:
            raise Exception(f"Timeout after {self.timeout}s")
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.error("embed.failed error=%s", str(e))
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Получить эмбеддинги для нескольких текстов (batch).
        
        Args:
            texts: Список текстов (могут быть пустые, они отфильтруются)
            
        Returns:
            List[List[float]]: Список векторов в том же порядке, что и входные тексты
        """
        if not texts:
            return []
        
        # Фильтруем пустые тексты, но сохраняем индексы
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts = [texts[i][:10000] for i in valid_indices]
        
        if not valid_texts:
            return [[] for _ in texts]
        
        payload = {
            "input": valid_texts,
            "model": self.model,
            "encoding_format": "float"
        }
        
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data:
                raise Exception("Invalid response: no 'data' field")
            
            # Сортируем по индексу
            embeddings = sorted(data["data"], key=lambda x: x.get("index", 0))
            vectors = [item.get("embedding", []) for item in embeddings]
            
            # Восстанавливаем порядок исходных текстов
            result = [[] for _ in texts]
            for idx, vec in zip(valid_indices, vectors):
                result[idx] = vec
            
            return result
            
        except Exception as e:
            logger.error("batch.embed.failed error=%s", str(e))
            raise
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()