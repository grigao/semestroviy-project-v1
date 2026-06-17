# worker_utils.py
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestion_parser import TelegramChannelParser
from config import settings

logger = logging.getLogger("worker_utils")


def load_workers(root: Path, filename: str = "workers.json") -> Dict[str, Any]:
    """Загружает конфигурацию воркеров"""
    workers_file = root / filename
    if not workers_file.exists():
        logger.warning("%s not found, using defaults", filename)
        return {}
    
    with open(workers_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_worker_config(root: Path, worker_name: str, filename: str = "workers.json") -> Optional[Dict[str, Any]]:
    """Получает конфигурацию конкретного воркера"""
    workers = load_workers(root, filename)
    return workers.get(worker_name)


def find_worker(root: Path, worker_name: str) -> Optional[Dict[str, Any]]:
    """Извлекает и валидирует параметры воркера"""
    worker_config = get_worker_config(root, worker_name)
    
    if not worker_config:
        available = list(load_workers(root).keys())
        logger.error(
            "Worker '%s' not found in workers.json. Available: %s",
            worker_name, available
        )
        return None
    
    # Извлекаем auth и основную структуру
    auth = worker_config.get("auth", {})
    device = auth.get("device", {})

    return {
        "session_name": auth.get("session_name"),
        "proxy": auth.get("proxy"),
        "device_model": device.get("device_model"),
        "system_version": device.get("system_version"),
        "app_version": device.get("app_version"),
        "channels": worker_config.get("channels")
    }


def init_parser(
    root: Path,
    session_name: str = "pyro_parser",
    proxy: dict = None,
    device_model: str = "Samsung Galaxy S23 Ultra",
    system_version: str = "Android 14",
    app_version: str = "10.3.2",
) -> TelegramChannelParser:
    """Создаёт парсер с кастомными параметрами"""
    if proxy:
        logger.info(
            "Using proxy: %s://%s:%s",
            proxy.get("scheme", "socks5"),
            proxy["hostname"],
            proxy["port"]
        )
    else:
        logger.info("No proxy")
    
    logger.info("Session: %s | Device: %s / %s", session_name, device_model, system_version)
    
    return TelegramChannelParser(
        api_id=settings.app_id,
        api_hash=settings.api_hash,
        session_name=session_name,
        proxy=proxy,
        device_model=device_model,
        system_version=system_version,
        app_version=app_version,
        session_workdir=str(root / "sessions"),
    )