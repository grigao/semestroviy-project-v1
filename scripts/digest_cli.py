#!/usr/bin/env python3
"""CLI для тестирования digest builder с гибкими параметрами"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)

from modules.digest_builder.builder import builder
from modules.digest_builder.contracts import DigestRequest


def print_usage():
    print("Usage:")
    print("  python scripts/digest_cli.py search <query> [max_items] [--categories cat1,cat2] [--sources src1,src2] [--period 24h]")
    print("  python scripts/digest_cli.py sub [categories] [period] [--sources src1,src2]")
    print()
    print("Examples:")
    print("  python scripts/digest_cli.py search 'погода' 5")
    print("  python scripts/digest_cli.py search 'russia' 5 --categories politics --sources moscowach --period 48h")
    print("  python scripts/digest_cli.py sub --sources habr_com --period 7d")
    print("  python scripts/digest_cli.py sub technology --sources habr_com")
    print("  python scripts/digest_cli.py sub technology 7d --sources habr_com")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print_usage()
    
    mode = sys.argv[1]
    
    if mode not in ("search", "sub"):
        print_usage()
    
    args = sys.argv[2:]
    categories = None
    sources = None
    period = None
    positional = []
    
    i = 0
    while i < len(args):
        if args[i] == "--categories":
            categories = args[i + 1].split(",")
            i += 2
        elif args[i] == "--sources":
            sources = args[i + 1].split(",")
            i += 2
        elif args[i] == "--period":
            period = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1
    
    if mode == "search":
        query = positional[0] if positional else "новости"
        max_items = int(positional[1]) if len(positional) > 1 else 5
        
        request = DigestRequest(
            query=query,
            categories=categories,
            sources=sources,
            period=period,
            max_items=max_items
        )
        result = builder.build_markdown(request)
        print(result)
    
    elif mode == "sub":
        # Категории из позиционных или --categories
        if positional:
            if positional[0] and positional[0] not in ("_", ""):
                categories = categories or positional[0].split(",")
            if len(positional) > 1 and positional[1] not in ("_", ""):
                period = period or positional[1]
        
        request = DigestRequest(
            categories=categories,
            sources=sources,
            period=period
        )
        result = builder.build_markdown(request)
        print(result)


if __name__ == "__main__":
    main()