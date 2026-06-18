import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import settings
import tiktoken

logger = logging.getLogger("c_blocks")


class ContentBlockBuilder:
    MAX_BLOCK_CHARS = 500
    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")

    BLOCK_SKIP_PATTERNS = [
        "мы в max",
        "не грузятся фото",
        "поддержите канал",
    ]

    def __init__(
        self,
        token_model: str = "cl100k_base",
        skip_patterns: Optional[List[str]] = None
    ):
        self.skip_patterns = skip_patterns or self.BLOCK_SKIP_PATTERNS
        self._skip_patterns_lower = [p.lower() for p in self.skip_patterns]
        self.encoder = tiktoken.get_encoding(token_model)
        self.max_block_tokens = int(self.MAX_BLOCK_CHARS / 1.5)

    def estimate_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def is_large_block(self, text: str) -> bool:
        if len(text) > self.MAX_BLOCK_CHARS:
            return True
        return self.estimate_tokens(text) > self.max_block_tokens

    @staticmethod
    def build_block_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _should_skip_block(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self._skip_patterns_lower)

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if "\n\n" in text:
            blocks = text.split("\n\n")
        elif "\n" in text:
            blocks = text.split("\n")
        else:
            blocks = [text]

        result = []
        for block in blocks:
            block = block.strip()
            if not block or self._should_skip_block(block):
                continue

            if self.is_large_block(block):
                sentences = re.split(self.SENTENCE_SPLIT_REGEX, block)
                result.extend([
                    s.strip() for s in sentences
                    if s.strip() and not self._should_skip_block(s.strip())
                ])
            else:
                result.append(block)

        return result

    def build_message_blocks(self, message: Dict[str, Any]) -> Dict[str, Any]:
        text = message.get("clean_text", "")
        if not text:
            message["content_blocks"] = []
            return message

        raw_blocks = self.split_text(text)

        content_blocks = [
            {
                "block_hash": self.build_block_hash(block_text),
                "position": position,
                "text": block_text,
                "char_count": len(block_text),
                "token_count": self.estimate_tokens(block_text),
                "embedding_status": "pending",
            }
            for position, block_text in enumerate(raw_blocks)
        ]

        message["content_blocks"] = content_blocks
        return message

    def process_file(self, input_file: Path, output_dir: Path) -> Path:
        """Обработать один файл (используется для снэпшотов)."""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("messages", [])
        for message in messages:
            self.build_message_blocks(message)

        output_path = output_dir / input_file.name.replace("_snapshot", "")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"messages": messages}, f, ensure_ascii=False, indent=2)

        return output_path

    def process_archive(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "output" / "blocks"

        # Обрабатываем все файлы, включая снэпшоты
        json_files = list(input_dir.glob("*.json"))
        logger.info("blocks.start files=%d input=%s", len(json_files), input_dir)

        output_paths = []
        total_blocks = 0
        total_skipped = 0

        for json_file in json_files:
            logger.info("-- @%s --------------------", json_file.stem)

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])
            file_blocks = 0
            file_skipped = 0

            for message in messages:
                old_blocks_count = len(message.get("content_blocks", []))
                self.build_message_blocks(message)
                new_blocks_count = len(message.get("content_blocks", []))

                file_blocks += new_blocks_count
                file_skipped += max(0, old_blocks_count - new_blocks_count)

                if settings.debug and new_blocks_count != old_blocks_count:
                    logger.debug(
                        "message.blocks id=%s blocks=%d skipped=%d",
                        message.get("id"), new_blocks_count, file_skipped
                    )

            metadata = data.setdefault("metadata", {})
            metadata["content_blocks_count"] = file_blocks
            metadata["content_blocks_skipped"] = file_skipped

            # Имя выходного файла без _snapshot
            output_name = json_file.name.replace("_snapshot", "")
            output_path = output_dir / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"metadata": metadata, "messages": messages},
                    f, ensure_ascii=False, indent=2
                )

            logger.info(
                "file.complete name=%s messages=%d blocks=%d skipped=%d",
                output_name, len(messages), file_blocks, file_skipped
            )
            output_paths.append(output_path)

            total_blocks += file_blocks
            total_skipped += file_skipped

        logger.info(
            "blocks.complete files=%d total=%d blocks=%d skipped=%d",
            len(output_paths), len(output_paths), total_blocks, total_skipped
        )
        return output_paths