import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import tiktoken

from .cache import EnrichmentCache
from .llm_client import LLMClient

from config import settings

logger = logging.getLogger("enricher")

class MessageEnricher:
    MODEL = "Qwen3.6"
    VERSION = "v1"
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    MAX_INPUT_TOKENS = 1000
    MAX_CONCURRENT_REQUESTS = 2

    def __init__(self, llm: Optional[LLMClient] = None, cache_path: Optional[Path] = None, batch_size: int = 20) -> None:
        self.llm = llm or LLMClient()
        if cache_path is None:
            cache_path = Path(__file__).parent.parent / "cache" / "enrichment_cache.json"
        self.cache = EnrichmentCache(cache_path)
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

    @classmethod
    def truncate_text(cls, text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
        if not text:
            return ""
        tokens = cls.TOKENIZER.encode(text)
        return text if len(tokens) <= max_tokens else cls.TOKENIZER.decode(tokens[:max_tokens])

    async def enrich_single_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        content_hash = message.get("content_hash")
        text = message.get("clean_text", "")
        
        default_enrichment = {
            "category": "other", 
            "keywords": [], 
            "entities": [],
            "model": self.MODEL,
            "version": self.VERSION
        }
        
        if not content_hash or not text:
            logger.warning("message.skipped id=%s reason=%s", message.get("id"), "no_content_hash" if not content_hash else "no_clean_text")
            message["enrichment"] = default_enrichment
            return message

        cached = self.cache.get(content_hash)
        if cached:
            # Проверяем версию кеша, если старая - обновляем
            if cached.get("version") != default_enrichment["version"] or cached.get("model") != default_enrichment["model"]:
                logger.info("cache.outdated hash=%s old_version=%s new_version=%s", 
                        content_hash, cached.get("version"), default_enrichment["version"])
            else:
                message["enrichment"] = cached
                return message

        truncated_text = self.truncate_text(text)
        async with self.semaphore:
            enrichment_data = await self.llm.enrich_text(truncated_text)
        
        enrichment = {
            **enrichment_data,
            "model": default_enrichment["model"],
            "version": default_enrichment["version"],
        }
        
        message["enrichment"] = enrichment
        self.cache.set(content_hash, enrichment)
        return message

    async def process_archive(self, input_dir: Path, output_dir: Optional[Path] = None) -> List[Path]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "output" / "enriched"
        
        json_files = list(input_dir.glob("*.json"))
        logger.info("enricher.start files=%d input=%s", len(json_files), input_dir)
        
        output_paths = []
        total_cache_hits = 0
        total_llm_calls = 0
        
        for json_file in json_files:
            logger.info("-- @%s --------------------", json_file.stem)

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            messages = data.get("messages", [])
            
            enriched_messages = []
            error_count = 0
            cache_hits = 0
            llm_calls = 0
            
            for i in range(0, len(messages), self.batch_size):
                batch = messages[i:i + self.batch_size]
                
                if settings.debug:
                    logger.debug("batch.start file=%s offset=%d size=%d", json_file.name, i, len(batch))
                
                for message in batch:
                    try:
                        content_hash = message.get("content_hash")
                        if content_hash and self.cache.get(content_hash):
                            cache_hits += 1
                            if settings.debug:
                                logger.debug("cache.hit hash=%s id=%s", content_hash[:8], message.get("id"))
                        else:
                            llm_calls += 1
                            if settings.debug:
                                logger.debug("llm.call id=%s", message.get("id"))
                        
                        enriched = await self.enrich_single_message(message)
                        enriched_messages.append(enriched)
                    except Exception:
                        logger.exception("message.failed id=%s", message.get("id"))
                        error_count += 1
                
                self.cache.save()
                
                if settings.debug:
                    logger.debug("batch.complete file=%s processed=%d", json_file.name, len(enriched_messages))
            
            output_data = {
                "metadata": {
                    **data.get("metadata", {}),
                    "enriched_at": datetime.utcnow().isoformat(),
                    "enriched_count": len(enriched_messages),
                    "enrichment_errors": error_count,
                    "cache_hits": cache_hits,
                    "llm_calls": llm_calls,
                },
                "messages": enriched_messages,
            }
            
            output_path = output_dir / json_file.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info("file.complete name=%s messages=%d errors=%d cached=%d llm=%d", 
                    json_file.name, len(enriched_messages), error_count, cache_hits, llm_calls)
            output_paths.append(output_path)
            
            total_cache_hits += cache_hits
            total_llm_calls += llm_calls
        
        await self.llm.close()
        logger.info("enricher.complete files=%d total=%d cached=%d llm=%d", 
                len(output_paths), len(output_paths), total_cache_hits, total_llm_calls)
        return output_paths


async def enrich_single_message_test(message: Dict[str, Any]) -> Dict[str, Any]:
    enricher = MessageEnricher(batch_size=2)
    enricher.cache._data = {}
    if isinstance(message, tuple):
        message = message[0]
    result = await enricher.enrich_single_message(message)
    await enricher.llm.close()
    return result


# --- Example usage
# Пример 1: Обработка одного сообщения
# test_msg = {
#     "id": 68862,
#     "date": "2026-06-11T14:05:09",
#     "urls": ["https://habr.com/ru/news/1045726"],
#     "post_link": "https://t.me/habr_com/68862",
#     "clean_text": "Финский стартап Donut Lab собрал $25 млн с 1300 мелких инвесторов под рассказы о твёрдотельной натриевой батарее с зарядкой за 5 минут. Внутри заявленного продукта оказался литий-ионный аккумулятор с плотностью 298 Вт·ч/кг. Химики из 20 лабораторий изучили начинку и доказали подмену через графики расширения анода.\nИнженеры лаборатории VTT прогнали элемент через стенды и зафиксировали напряжение 3,7-3,8 вольт при 50% заряда. Натриевые ячейки по законам физики не выдают больше 3,5 вольт. Графитовый анод при зарядке показал излом расширения в диапазоне 50-70%. Ионы натрия слишком громоздкие для проникновения в слоистую структуру графита.\nСледы технологии привели к немецкой компании CT Coatings. Этот подрядчик держит патенты на трафаретную печать по тротуарной плитке и обложкам ресторанных меню. Немцы обещали напечатать натриевую батарею, но отгрузили финнам стандартный аккумулятор в пакете. На технических встречах авторы разработки всерьёз приравнивали литий к редкоземельным металлам.\nРуководство стартапа до последнего выпускало отчёты о мощности зарядки свыше 100 киловатт. Обещанный к релизу в I квартале 2026 года серийный электромотоцикл оказался тестовым прототипом для внутреннего автопарка. Директор Марко Лехтимяки признал использование совсем других элементов питания в этой технике. Сейчас документацию компании с оценкой в $1,25 млрд изучает финская полиция.",
#     "hashtags": [],
#     "mentions": [],
#     "content_hash": "b194dc6193cf3ab4685aed730741ef9abddc75e0ab28472deb7f7e762954013c"
# }

# result = await enrich_single_message_test(test_msg)
# print(json.dumps(result, ensure_ascii=False, indent=2))
from ..content_blocks.block_builder import ContentBlockBuilder

async def main():    
    BASE_DIR = Path(__file__).resolve().parent
    
    input_dir = BASE_DIR / ".." / "output" / "processed"
    enriched_dir = BASE_DIR / ".." / "output" / "enriched"
    blocks_dir = BASE_DIR / ".." / "output" / "blocks"
    
    # Шаг 1: Обогащение всех файлов
    enricher = MessageEnricher(batch_size=10)
    enriched_paths = await enricher.process_archive(
        input_dir=input_dir.resolve(),
        output_dir=enriched_dir.resolve(),
    )
    
    # Шаг 2: Построение блоков для всех обогащённых файлов
    builder = ContentBlockBuilder()
    blocks_paths = builder.process_archive(
        input_dir=enriched_dir.resolve(),
        output_dir=blocks_dir.resolve(),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main())