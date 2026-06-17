"""Генерация текста дайджеста через LLM"""
import json
import re
import logging
from modules.digest_builder.contracts import DigestResponse

logger = logging.getLogger("digest.generator")


class DigestGenerator:
    """Генератор финального текста дайджеста через LLM"""
    
    def __init__(self, llm_client=None):
        if llm_client is None:
            from modules.digest_builder.llm_client import DigestLLMClient
            self.llm = DigestLLMClient()
        else:
            self.llm = llm_client
    
    def _build_prompt(self, response: DigestResponse) -> str:
        items_text = []
        for i, item in enumerate(response.items, 1):
            items_text.append(f"{i}. {item.text[:300]}")
        
        items_block = "\n".join(items_text)
        query_line = f"По запросу: «{response.query}»\n" if response.query else ""
        mode_line = "поисковый" if response.mode == "search" else "обзорный"
        
        return f"""Составь краткий {mode_line} дайджест на русском языке.

{query_line}
Новости (цифра в начале — номер новости):
{items_block}

Верни ТОЛЬКО JSON, без markdown-разметки и пояснений:
{{
  "title": "Заголовок дайджеста",
  "items": [
    {{
      "summary": "Краткая аннотация (1-2 предложения)",
      "source_indices": [1]
    }}
  ]
}}

Правила:
- Опирайся только на предоставленные тексты.
- Если новость нерелевантна запросу или слишком короткая/неинформативная — ПРОПУСТИ её (не включай в source_indices).
- Для обзорного дайджеста НЕ группируй новости, каждая идёт отдельно.
- Для поискового дайджеста можешь группировать похожие.
- Пиши грамотно на русском языке.
- Возвращай ТОЛЬКО JSON."""
    
    def _extract_json(self, text: str) -> str:
        """Извлекает JSON из текста (может быть внутри reasoning)."""
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0)
        return text
    
    def generate(self, response: DigestResponse) -> str:
        prompt = self._build_prompt(response)
        logger.info("Generating digest: %d items", len(response.items))
        
        try:
            result = self.llm.generate(prompt)
            result = self._extract_json(result)
            logger.info("Digest generated, length=%d chars", len(result))
            return result
        except Exception as e:
            logger.error("Digest generation failed: %s", str(e))
            items_json = [
                {"summary": item.text[:200], "source_indices": [i + 1]}
                for i, item in enumerate(response.items)
            ]
            return json.dumps({
                "title": "Дайджест",
                "items": items_json
            }, ensure_ascii=False, indent=2)