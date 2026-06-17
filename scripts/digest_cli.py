#!/usr/bin/env python3
"""CLI для тестирования digest builder"""
import sys
import logging
from pathlib import Path
import signal
import sys

# Ctrl+C → моментальный выход
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)

from modules.digest_builder.builder import builder
from modules.digest_builder.contracts import DigestRequest


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/digest_cli.py search 'query' [max_items=5]")
        print("  python scripts/digest_cli.py sub [categories] [period=24h]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else "новости"
        max_items = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        request = DigestRequest(query=query, max_items=max_items)
        result = builder.build_markdown(request)
        print(result)
        
    elif mode == "sub":
        categories = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
        period = sys.argv[3] if len(sys.argv) > 3 else "24h"
        
        request = DigestRequest(categories=categories, period=period)
        result = builder.build_markdown(request)
        print(result)
    
    else:
        print(f"Неизвестный режим: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()