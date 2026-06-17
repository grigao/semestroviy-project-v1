"""LLM-клиент для генерации дайджестов через LM Studio"""
import logging
from typing import Any, Dict, Optional
import httpx

from config import settings

logger = logging.getLogger("digest.llm")

DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "source_indices": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["summary", "source_indices"],
                "additionalProperties": False
            }
        }
    },
    "required": ["title", "items"],
    "additionalProperties": False
}


class DigestLLMClient:
    """Client for local Qwen3 via LM Studio for digest generation."""

    def __init__(
        self,
        base_url: str = None,
        model: str = "qwen3-27b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or settings.lmstudio_base_url).rstrip("/") + "/v1"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """Extract text from response, falling back to reasoning_content."""
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Missing choices[0].message: {e}")

        content = message.get("content", "")
        if content and content.strip():
            return content

        reasoning = message.get("reasoning_content", "")
        if reasoning and reasoning.strip():
            logger.info("Using reasoning_content fallback (%d chars)", len(reasoning))
            return reasoning

        raise ValueError("Both content and reasoning_content are empty")

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты — составитель дайджестов. Проанализируй новости и верни строгий JSON по схеме.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "digest_output",
                    "strict": True,
                    "schema": DIGEST_SCHEMA,
                },
            },
            "thinking_budget": 0,
            "extra_body": {"thinking_budget": 0},
        }

    def generate(self, prompt: str) -> str:
        """Генерирует дайджест."""
        payload = self._build_payload(prompt)

        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return self._extract_content(data)
            except Exception as e:
                logger.error("LLM attempt %d failed: %s", attempt + 1, str(e))
                if attempt == self.max_retries - 1:
                    raise
        return ""