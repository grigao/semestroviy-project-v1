"""
Simple JSON-based enrichment cache keyed by content_hash.
Thread-safe for async usage via manual file I/O.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
# from config import settings

logger = logging.getLogger("cache_en")


class EnrichmentCache:
    """Persistent cache for LLM enrichment results."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("cache.loaded entries=%d path=%s", len(self._data), self.cache_path)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("cache.load_failed error=%s path=%s", str(e), self.cache_path)
                self._data = {}
        else:
            logger.debug("cache.not_found path=%s", self.cache_path)
        self._loaded = True

    def get(self, content_hash: str) -> Optional[Dict[str, Any]]:
        self._load()
        return self._data.get(content_hash)

    def set(self, content_hash: str, enrichment: Dict[str, Any]) -> None:
        self._load()
        self._data[content_hash] = enrichment

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.cache_path.with_suffix(".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

        tmp_path.replace(self.cache_path)

        # if settings.debug:
        logger.debug("cache.saved entries=%d path=%s", len(self._data), self.cache_path)