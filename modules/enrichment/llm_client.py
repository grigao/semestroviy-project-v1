import json
import logging
import re
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("lmclient")

ALLOWED_CATEGORIES = [
    "technology", "ai", "security", "science", "business", "finance",
    "politics", "society", "news", "entertainment", "sports", "other",
    "health", "education", "ecology", "transport", "space",
    "law", "energy", "agriculture", "media", "gaming", "crypto",
]

ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ALLOWED_CATEGORIES,
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["category", "keywords", "entities"],
    "additionalProperties": False,
}


class LLMClient:
    """Async client for local Qwen3 via LM Studio."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen3-27b",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
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

    def _build_payload(self, text: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты — анализатор контента из мессенджера. Проанализируй пост и верни строгий JSON по схеме.",
                },
                {"role": "user", "content": f"Пост:\n\n{text}"},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "telegram_analysis",
                    "strict": True,
                    "schema": ENRICHMENT_SCHEMA,
                },
            },
            "thinking_budget": 0,
            "extra_body": {"thinking_budget": 0},
        }

    async def enrich_text(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            logger.error("Empty text")
            return {"category": "other", "keywords": [], "entities": []}

        payload = self._build_payload(text)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                raw_output = self._extract_content(data)
                parsed = self._parse_json(raw_output)
                validated = self._validate_enrichment(parsed)

                logger.debug(
                    "Enriched: category=%s, keywords=%d, entities=%d",
                    validated["category"],
                    len(validated["keywords"]),
                    len(validated["entities"]),
                )
                return validated

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.warning("Attempt %d failed: %s", attempt, last_error)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = str(e)
                logger.warning("Attempt %d failed: %s", attempt, last_error)
            except httpx.TimeoutException:
                last_error = "timeout"
                logger.warning("Attempt %d failed: %s", attempt, last_error)
            except Exception as e:
                last_error = f"Unexpected error: {type(e).__name__}: {str(e)}"
                logger.warning("Attempt %d failed: %s", attempt, last_error)

            if attempt < self.max_retries:
                await __import__("asyncio").sleep(2 ** attempt)

        logger.error("All %d attempts failed. Last error: %s", self.max_retries, last_error)
        return {"category": "other", "keywords": [], "entities": []}

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        cleaned = raw.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found")

        cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)

    @staticmethod
    def _validate_enrichment(data: Dict[str, Any]) -> Dict[str, Any]:
        category = data.get("category", "other")
        if category not in ALLOWED_CATEGORIES:
            category = "other"

        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k).lower().strip() for k in keywords if k][:7]

        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        entities = [str(e).strip() for e in entities if e][:5]

        return {
            "category": category,
            "keywords": keywords,
            "entities": entities,
        }
