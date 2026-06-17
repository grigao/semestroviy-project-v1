import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional, Callable

from pyrogram import Client
from pyrogram.types import Message
from pyrogram.errors import FloodWait, RPCError
from pyrogram.enums import MessageEntityType

logger = logging.getLogger("ingestion_parser")

logging.getLogger("pyrogram").setLevel(logging.CRITICAL)
logging.getLogger("pyrogram.session.session").setLevel(logging.CRITICAL)
logging.getLogger("pyrogram.dispatcher").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# -----------------------------
# Retry decorator (pure)
# -----------------------------

def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    def decorator(fn: Callable):
        async def wrapper(*args, **kwargs):
            attempt = 0

            while True:
                try:
                    return await fn(*args, **kwargs)

                except FloodWait as e:
                    attempt += 1
                    if attempt > max_retries:
                        return {"error": f"FloodWait exceeded: {e.value}s"}

                    wait_time = e.value
                    logger.warning(
                        "FloodWait=%ss attempt=%s/%s",
                        wait_time,
                        attempt,
                        max_retries
                    )
                    await asyncio.sleep(wait_time)

                except (RPCError, ConnectionError, TimeoutError) as e:
                    attempt += 1
                    if attempt > max_retries:
                        return {"error": f"RPC/Network error: {str(e)}"}

                    wait_time = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Retryable error=%s attempt=%s/%s wait=%ss",
                        str(e),
                        attempt,
                        max_retries,
                        wait_time
                    )
                    await asyncio.sleep(wait_time)

        return wrapper

    return decorator


# -----------------------------
# Parser (stateless core)
# -----------------------------
def extract_urls(message: Message) -> list[str]:
    urls: set[str] = set()

    entities = (message.entities or []) + (message.caption_entities or [])

    for entity in entities:
        if entity.type == MessageEntityType.TEXT_LINK and entity.url:
            urls.add(entity.url)

    return list(urls)

class TelegramChannelParser:
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str = "pyro_parser",
        proxy: Optional[dict] = None,
        device_model: str = "Samsung Galaxy S23 Ultra",
        system_version: str = "Android 14",
        app_version: str = "10.3.2",
        session_workdir: str = "sessions",
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.proxy = proxy

        self.device_model = device_model
        self.system_version = system_version
        self.app_version = app_version

        self.session_workdir = session_workdir

        self._client: Optional[Client] = None

    # -----------------------------
    # client lifecycle (only stateful part)
    # -----------------------------

    async def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(
                name=self.session_name,
                api_id=self.api_id,
                api_hash=self.api_hash,
                proxy=self.proxy,
                workdir=self.session_workdir,
                device_model=self.device_model,
                system_version=self.system_version,
                app_version=self.app_version,
            )
            await self._client.start()

        return self._client

    async def close(self):
        if self._client:
            await self._client.stop()
            self._client = None

    # -----------------------------
    # get urls
    # -----------------------------

    @staticmethod
    def _extract_urls_from_entities(message: Message, text: str) -> list[str]:
        urls = []
        entities = getattr(message, "entities", None) or getattr(message, "caption_entities", None)
        
        if not entities or not text:
            return urls
        
        seen = set()
        
        for ent in entities:
            url = None
            
            if ent.type == "text_link":
                url = getattr(ent, "url", None)
            elif ent.type == "url":
                # Plain URL — вырезаем подстроку из текста
                url = text[ent.offset : ent.offset + ent.length]
            
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        
        return urls
    
    # -----------------------------
    # message normalization
    # -----------------------------

    @staticmethod
    def _format_message(message: Message, clean: bool = True) -> dict[str, Any]:
        text = message.text or message.caption or ""

        link = None
        if message.chat and message.chat.username:
            link = f"https://t.me/{message.chat.username}/{message.id}"

        base = {
            "id": message.id,
            "date": message.date.isoformat() if message.date else None,
            "text": text[:4096] if text else "",
            "urls": extract_urls(message),
            "post_link": link,
        }

        if not clean:
            base["has_media"] = bool(message.media)

        return base

    # -----------------------------
    # empty text checker
    # -----------------------------

    @staticmethod
    def _is_valid_message(message: Message) -> bool:
        text = message.text
        caption = message.caption

        has_text = isinstance(text, str) and text.strip() != ""
        has_caption = isinstance(caption, str) and caption.strip() != ""
        has_media = bool(message.media)
        
        # Проверяем, является ли сообщение частью альбома
        is_album = bool(message.media_group_id)

        # Пропускаем системные сообщения без контента
        if not has_text and not has_caption and not has_media:
            return False
        
        # Если это часть альбома, но без подписи — пропускаем
        if is_album and not has_text and not has_caption:
            return False

        return True

    # -----------------------------
    # CORE: deterministic async iterator
    # -----------------------------

    async def iter_channel_messages(
        self,
        channel: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        clean: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:

        client = await self._get_client()

        channel_username = channel.lstrip("@").split("t.me/")[-1]
        chat = await client.get_chat(channel_username)

        async for message in client.get_chat_history(chat.id):
            if not isinstance(message, Message):
                continue

            msg_date = message.date

            if end_date and msg_date > end_date:
                continue
            if start_date and msg_date < start_date:
                break

            if not self._is_valid_message(message):
                logger.warning(
                    "SKIP_EMPTY_MSG id=%s date=%s has_text=%s has_caption=%s has_media=%s",
                    message.id,
                    msg_date,
                    bool(message.text),
                    bool(message.caption),
                    bool(getattr(message, "media", None)),
                )
                continue

            yield self._format_message(message, clean=clean)


    # -----------------------------
    # ingestion wrapper (materializer)
    # -----------------------------

    @retry_async(max_retries=3)
    async def fetch_channel(
        self,
        channel: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        clean: bool = True,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:

        client = await self._get_client()

        chat = await client.get_chat(channel.lstrip("@"))

        messages = []
        oldest = None
        newest = None

        async for msg in self.iter_channel_messages(
            channel=channel,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
        ):

            messages.append(msg)

            dt = datetime.fromisoformat(msg["date"])

            oldest = dt if oldest is None else min(oldest, dt)
            newest = dt if newest is None else max(newest, dt)

            if limit and len(messages) >= limit:
                break

        return {
            "source": {
                "id": chat.id,
                "username": chat.username,
                "title": chat.title,
                "link": f"https://t.me/{chat.username}" if chat.username else None,
            },
            "messages": messages,
            "meta": {
                "fetched_messages": len(messages),
                "oldest_message_date": oldest.isoformat() if oldest else None,
                "newest_message_date": newest.isoformat() if newest else None,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
            },
        }
