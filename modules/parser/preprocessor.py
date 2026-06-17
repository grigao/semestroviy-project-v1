import hashlib
import json
import logging
import re

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

URL_REGEX = re.compile(r"https?://[^\s]+|t\.me/[^\s]+")
HASHTAG_REGEX = re.compile(r"#\w+")
MENTION_REGEX = re.compile(r"@\w+")
WHITESPACE_REGEX = re.compile(r"[^\S\n]+")

# -----------------------------
# helpers
# -----------------------------

def normalize_text(text: str) -> str:
    text = WHITESPACE_REGEX.sub(" ", text)

    lines = [line.strip() for line in text.splitlines()]

    return "\n".join(line for line in lines if line)


def extract_urls(text: str) -> list[str]:
    return URL_REGEX.findall(text)


def extract_hashtags(text: str) -> list[str]:
    return HASHTAG_REGEX.findall(text)


def extract_mentions(text: str) -> list[str]:
    return MENTION_REGEX.findall(text)


def build_content_hash(text: str) -> str:
    text_without_urls = URL_REGEX.sub("", text)

    normalized = normalize_text(text_without_urls)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# -----------------------------
# processor itself
# -----------------------------

def process_message(message: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = message["text"]

    processed_text = normalize_text(raw_text)

    urls_ent = message.get("urls", [])
    urls_regex = extract_urls(processed_text)
    all_urls = list(dict.fromkeys(urls_ent + urls_regex))

    return {
        **{k: v for k, v in message.items() if k != "text"},

        "clean_text": processed_text,
        # "is_processed": bool(processed_text != raw_text),

        # Записываем объединенный список строк
        "urls": all_urls,
        "hashtags": extract_hashtags(processed_text),
        "mentions": extract_mentions(processed_text),

        "content_hash": build_content_hash(processed_text),
    }

class MessageProcessor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_archive(
        self,
        archive_path: Path,
    ) -> Path:
        """
        raw archive -> processed archive

        Returns:
            Path to processed file
        """

        if not archive_path.exists():
            raise FileNotFoundError(archive_path)

        with open(archive_path, "r", encoding="utf-8") as f:
            archive = json.load(f)

        messages = archive.get("messages", [])

        processed_messages = [
            process_message(message)
            for message in messages
        ]

        processed_messages = [
            message
            for message in processed_messages
            if message is not None
        ]

        processed_archive = {
            "metadata": {
                **archive.get("metadata", {}),
                "updated_at": datetime.now().replace(microsecond=0).isoformat(),
                "processed_messages": len(processed_messages),
            },
            "messages": processed_messages,
        }

        output_path = self.output_dir / archive_path.name

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                processed_archive,
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "processor.done file=%s messages=%s",
            archive_path.name,
            len(processed_messages),
        )

        return output_path
