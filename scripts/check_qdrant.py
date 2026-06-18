"""Полная отладка digest builder с подробным логом"""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)

from modules.digest_builder.builder import builder, BLOCKS_DIR
from modules.digest_builder.contracts import DigestRequest

print(f"BLOCKS_DIR: {BLOCKS_DIR}")
print(f"exists: {BLOCKS_DIR.exists()}")
print()

req = DigestRequest(sources=["sportclubithub"])
response = builder.build(req)

print(f"\nItems: {len(response.items)}")
for i, item in enumerate(response.items, 1):
    text_preview = item.text[:80] if item.text else "*** ТЕКСТ НЕ ЗАГРУЖЕН ***"
    print(f"{i}. [{item.source}] {text_preview}")
    print(f"   hash: {item.block_hash[:32]}...")
    print(f"   link: {item.post_link}")
    print()