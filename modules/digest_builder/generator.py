"""Генерация текста дайджеста через LLM"""
import json
import re
import logging
from datetime import datetime
from modules.digest_builder.contracts import DigestResponse
from modules.digest_builder.prompts import build_digest_prompt

logger = logging.getLogger("digest.generator")


class DigestGenerator:
    def __init__(self, llm_client=None):
        if llm_client is None:
            from modules.digest_builder.llm_client import DigestLLMClient
            self.llm = DigestLLMClient()
        else:
            self.llm = llm_client
    
    def _build_prompt(self, response: DigestResponse) -> str:
        items_text = "\n".join(
            f"{i}. {item.text[:300]}"
            for i, item in enumerate(response.items, 1)
        )
        today = datetime.utcnow().strftime("%d.%m.%Y")
        
        return build_digest_prompt(
            items_text=items_text,
            query=response.query,
            today=today
        )
    
    def _extract_json(self, text: str) -> str:
        match = re.search(r'\{[\s\S]*\}', text)
        return match.group(0) if match else text
    
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
            return json.dumps({
                "title": "Дайджест",
                "items": [
                    {"summary": item.text[:200], "source_indices": [i + 1]}
                    for i, item in enumerate(response.items)
                ]
            }, ensure_ascii=False, indent=2)