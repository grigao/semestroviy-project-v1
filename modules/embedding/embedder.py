import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from config import settings
from modules.embedding.services.embed_service import EmbeddingService
from modules.qdrant.collections import init_collections
from modules.qdrant.uploader import QdrantUploader
from modules.qdrant.models import BlockPayload

logger = logging.getLogger("embedding")


def extract_source_from_file_path(file_path: Path) -> str:
    """Извлекает имя источника из имени файла (habr_com.json → habr_com)"""
    return file_path.stem

async def process_blocks_archive(input_dir: Path) -> int:
    block_files = list(input_dir.glob("*.json"))
    if not block_files:
        logger.warning("blocks.input.empty path=%s", input_dir)
        return 0

    logger.info("embedding.start files=%d input=%s", len(block_files), input_dir)

    all_blocks: List[Dict[str, Any]] = []
    stats = {"total": 0, "pending": 0, "completed": 0, "skipped": 0}

    for block_file in block_files:
        source = extract_source_from_file_path(block_file)
        try:
            with open(block_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "messages" in data:
                for message in data.get("messages", []):
                    enrichment = message.get("enrichment", {})
                    for block in message.get("content_blocks", []):
                        stats["total"] += 1
                        if block.get("embedding_status") != "completed":
                            block["source"] = source
                            block["message_id"] = str(message.get("id", ""))
                            block["published_at"] = message.get("date")
                            block["category"] = enrichment.get("category")
                            block["keywords"] = enrichment.get("keywords")
                            block["post_link"] = message.get("post_link")
                            all_blocks.append(block)
                            stats["pending"] += 1
                        else:
                            stats["completed"] += 1
            else:
                logger.warning("blocks.unknown_format file=%s", block_file.name)
                stats["skipped"] += 1
        except Exception as e:
            logger.error("blocks.read_failed file=%s error=%s", block_file.name, str(e))
            stats["skipped"] += 1

    logger.info(
        "blocks.scan total=%d pending=%d completed=%d skipped=%d",
        stats["total"], stats["pending"], stats["completed"], stats["skipped"]
    )

    if not all_blocks:
        logger.info("embedding.nothing_to_process")
        return 0

    embedding_service = EmbeddingService()
    init_collections()
    qdrant_uploader = QdrantUploader()

    total_success = 0
    total_failed = 0
    uploaded_count = 0

    try:
        batch_size = settings.embedding_batch_size
        total_batches = (len(all_blocks) + batch_size - 1) // batch_size

        for i in range(0, len(all_blocks), batch_size):
            batch_num = i // batch_size + 1
            batch = all_blocks[i:i + batch_size]
            logger.info("batch.start %d/%d blocks=%d", batch_num, total_batches, len(batch))

            batch_result = await embedding_service.generate_embeddings_batch(batch)
            total_success += batch_result.total_success
            total_failed += batch_result.total_failed

            for block, result in zip(batch, batch_result.results):
                if not result.success:
                    logger.error(
                        "block.failed hash=%s error=%s",
                        result.block_hash[:8], result.error_message
                    )
                    continue

                # Конвертация даты в timestamp
                published_at = block.get("published_at") or block.get("date")
                if published_at and isinstance(published_at, str):
                    try:
                        published_at = datetime.fromisoformat(published_at).timestamp()
                    except (ValueError, TypeError):
                        published_at = None

                payload = BlockPayload(
                    block_hash=result.block_hash,
                    message_id=block.get("message_id", ""),
                    source=block.get("source", "unknown"),
                    channel=block.get("source", "unknown"),
                    category=block.get("category"),
                    keywords=block.get("keywords"),
                    published_at=published_at,
                    token_count=block.get("token_count"),
                    embedding_model=result.metadata.get("embedding_model", ""),
                    embedding_version=result.metadata.get("embedding_version", ""),
                    embedding_status=result.metadata.get("embedding_status", ""),
                    error_message=result.error_message
                ).model_dump()

                qdrant_uploader.upsert({
                    "id": result.block_hash,
                    "vector": result.vector,
                    "payload": payload
                })
                uploaded_count += 1

            logger.info(
                "batch.complete %d/%d success=%d failed=%d",
                batch_num, total_batches,
                batch_result.total_success, batch_result.total_failed
            )

        logger.info(
            "embedding.complete total=%d success=%d failed=%d uploaded=%d",
            len(all_blocks), total_success, total_failed, uploaded_count
        )
    finally:
        await embedding_service.close()

    return uploaded_count


async def main():
    """Основная функция запуска"""
    logger.info("embedding.layer.start")

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    input_dir = BASE_DIR / "modules" / "output" / "blocks"

    logger.info(
        "embedding.config model=%s dim=%d batch=%d",
        settings.embedding_model,
        settings.embedding_dimension,
        settings.embedding_batch_size
    )

    if not input_dir.exists():
        logger.error("embedding.input_not_found path=%s", input_dir)
        logger.info("embedding.hint Please run content block builder first")
        return

    uploaded = await process_blocks_archive(input_dir)

    logger.info("embedding.layer.finish uploaded=%d", uploaded)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main())