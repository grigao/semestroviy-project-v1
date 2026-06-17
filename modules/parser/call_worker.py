import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from .ingestion_parser import TelegramChannelParser
from .storage import (
    archive_path,
    load_archive,
    save_archive,
    merge_by_id,
    get_coverage,
    update_coverage,
)
from .worker_utils import find_worker, init_parser
from .preprocessor import MessageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("call_worker")


# -----------------------------
# core ingestion logic
# -----------------------------

async def fetch(
    parser: TelegramChannelParser,
    channel: str,
    start: datetime,
    end: datetime,
) -> Dict[str, Any]:
    return await parser.fetch_channel(
        channel=channel,
        start_date=start,
        end_date=end,
        clean=True,
    )


async def sync_channel(
    parser: TelegramChannelParser,
    channel: str,
    start: datetime,
    end: datetime,
    input_path: Path,
):
    """Инкрементальная синхронизация канала"""
    path = archive_path(input_path, channel)
    archive = load_archive(path)
    
    meta = archive.get("metadata", {})
    messages = archive.get("messages", [])
    
    coverage_start, coverage_end = get_coverage(meta)
    
    logger.info(
        "archive.loaded messages=%d coverage=%s -> %s",
        len(messages),
        coverage_start.isoformat() if coverage_start else None,
        coverage_end.isoformat() if coverage_end else None,
    )
    
    # Первый запуск
    if not coverage_start or not coverage_end:
        logger.info("initial.ingestion range=%s→%s", start, end)
        
        result = await fetch(parser, channel, start, end)
        
        if "error" in result:
            logger.error("parser.error=%s", result["error"])
            return
        
        archive = {
            "metadata": {
                "coverage_start": start.isoformat(),
                "coverage_end": end.isoformat(),
                "updated_at": datetime.now().replace(microsecond=0).isoformat(),
            },
            "messages": result["messages"],
        }
        
        save_archive(path, archive)
        logger.info("initial.done messages=%d", len(result["messages"]))
        return
    
    # Инкрементальная загрузка
    tasks = []
    
    if start < coverage_start:
        tasks.append(
            fetch(parser, channel, start, coverage_start - timedelta(microseconds=1))
        )
    
    if end > coverage_end:
        tasks.append(
            fetch(parser, channel, coverage_end + timedelta(microseconds=1), end)
        )
    
    if not tasks:
        logger.info("up_to_date")
        return
    
    results = await asyncio.gather(*tasks)
    
    new_messages = []
    for r in results:
        if "messages" in r:
            new_messages.extend(r["messages"])
    
    merged, added, dup = merge_by_id(messages, new_messages)
    
    archive["messages"] = merged
    archive["metadata"] = update_coverage(meta, new_messages)
    
    save_archive(path, archive)
    
    logger.info(
        "sync.done added=%d dup=%d total=%d",
        added, dup, len(merged)
    )


# -----------------------------
# entrypoint
# -----------------------------

async def main():
    PARSER_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = PARSER_DIR.parent.parent

    WORKER_NAME = "worker_warp"
    
    params = find_worker(PROJECT_ROOT, WORKER_NAME)
    if not params:
        return
    
    # Подготовка запуска
    now = datetime.now().replace(microsecond=0)
    start = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0)
    
    parser = init_parser(
        root=PARSER_DIR,
        session_name=params["session_name"],
        proxy=params["proxy"],
        device_model=params["device_model"],
        system_version=params["system_version"],
        app_version=params["app_version"],
    )
    processor = MessageProcessor(
        output_dir= Path(PARSER_DIR).resolve().parent / "output" / "processed"
    )
    logger.info("Worker: %s", WORKER_NAME)
    logger.info("Date range: %s -> %s", start, now)
    
    try:
        for channel in params["channels"]:
            logger.info("-- %s --------------------", channel)

            await sync_channel(
                parser,
                channel,
                start,
                now,
                PARSER_DIR,
            )

            channel_name = channel.lstrip("@")
            raw_archive = (
                Path(PARSER_DIR).resolve().parent / "output" / "raw" / f"{channel_name}.json"
            )
            processor.process_archive(raw_archive)

    finally:
        await parser.close()
    

if __name__ == "__main__":
    asyncio.run(main())