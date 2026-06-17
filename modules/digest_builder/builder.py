"""Оркестратор сборки дайджеста"""
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from modules.qdrant.search import QdrantSearch
from modules.embedding.providers.lmstudio import LMStudioEmbeddingProvider
from modules.digest_builder.contracts import DigestRequest, DigestItem, DigestResponse
from modules.digest_builder.filters import build_filter
from modules.digest_builder.reranker import Reranker
from modules.digest_builder.ranker import rank_candidates
from modules.digest_builder.generator import DigestGenerator
from modules.digest_builder.formatter import format_digest_markdown
from modules.digest_builder.config import config

logger = logging.getLogger("digest.builder")

BLOCKS_DIR = Path(__file__).resolve().parent.parent.parent / "modules" / "output" / "blocks"


class DigestBuilder:
    
    def __init__(self):
        self.searcher = QdrantSearch()
        self.reranker = Reranker()
        self.generator = DigestGenerator()
        self._embedder = None
    
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = LMStudioEmbeddingProvider()
        return self._embedder
    
    def _get_query_vector(self, query: Optional[str]) -> Optional[List[float]]:
        if query:
            import asyncio
            return asyncio.run(self.embedder.embed(query))
        return None
    
    def _load_texts(self, block_hashes: set) -> tuple[Dict[str, str], Dict[str, str]]:
        """Подгружает тексты и ссылки из blocks/*.json"""
        texts = {}
        links = {}
        found = set()
        
        if not BLOCKS_DIR.exists():
            logger.warning("Blocks dir not found: %s", BLOCKS_DIR)
            return texts, links
        
        for file_path in BLOCKS_DIR.glob("*.json"):
            if found >= block_hashes:
                break
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "messages" in data:
                    for msg in data["messages"]:
                        msg_link = msg.get("post_link", "")
                        for block in msg.get("content_blocks", []):
                            h = block.get("block_hash")
                            if h in block_hashes and h not in found:
                                texts[h] = block.get("text", "")
                                links[h] = msg_link
                                found.add(h)
            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path.name, e)
                continue
        
        logger.info("Loaded %d texts from blocks files", len(texts))
        return texts, links
    
    def build(self, request: DigestRequest) -> DigestResponse:
        if not request.query and request.categories:
            request.query = f"новости по теме: {' '.join(request.categories)}"
                
        mode = "search" if request.query else "subscription"

        logger.info("Building digest: mode=%s, query=%s", mode, request.query)
        
        query_vector = self._get_query_vector(request.query)
        qdrant_filter = build_filter(request)
        
        if query_vector:
            candidates = self.searcher.search(
                vector=query_vector, limit=config.max_candidates, filter=qdrant_filter
            )
        else:
            candidates = self.searcher.search(
                vector=[0.0] * 1024, limit=config.max_candidates, filter=qdrant_filter
            )
        
        if not candidates:
            return DigestResponse(
                query=request.query, mode=mode, items=[],
                generated_at=datetime.utcnow().isoformat()
            )
        
        logger.info("Candidates: %d", len(candidates))
        
        candidate_dicts = []
        block_hashes = set()
        for c in candidates:
            p = c["payload"]
            candidate_dicts.append(p)
            block_hashes.add(p.get("block_hash", ""))
        
        texts_map, links_map = self._load_texts(block_hashes)
        
        # Дедупликация по message_id (оставляем первый блок из каждого сообщения)
        seen_msg_ids = set()
        unique_candidates = []
        unique_dicts = []
        for c, d in zip(candidates, candidate_dicts):
            mid = d.get("message_id", "")
            if mid not in seen_msg_ids:
                seen_msg_ids.add(mid)
                unique_candidates.append(c)
                unique_dicts.append(d)

        candidates = unique_candidates
        candidate_dicts = unique_dicts
        logger.info("After message dedup: %d candidates", len(candidates))
        
        documents = []
        vectors = []
        for c in candidate_dicts:
            h = c.get("block_hash", "")
            text = texts_map.get(h, "")
            documents.append(text)
            vectors.append(c.get("vector", [0.0] * 1024))
        
        if request.query and documents:
            ranked = self.reranker.rerank(request.query, documents)
            reranker_scores = [0.0] * len(documents)
            for idx, score in ranked:
                reranker_scores[idx] = float(score)
        else:
            reranker_scores = [c.get("score", 0.0) for c in candidates]
        
        selected_indices, final_scores = rank_candidates(
            candidate_dicts, query_vector or [0.0] * 1024,
            reranker_scores, vectors, mode, request.max_items
        )
        
        items = []
        for idx in selected_indices:
            c = candidate_dicts[idx]
            h = c.get("block_hash", "")
            items.append(DigestItem(
                block_hash=h,
                message_id=c.get("message_id", ""),
                source=c.get("source", "unknown"),
                category=c.get("category"),
                keywords=c.get("keywords"),
                published_at=c.get("published_at"),
                text=texts_map.get(h, ""),
                score=reranker_scores[idx],
                post_link=links_map.get(h)
            ))
        
        return DigestResponse(
            query=request.query, mode=mode, items=items,
            generated_at=datetime.utcnow().isoformat()
        )
    
    def build_markdown(self, request: DigestRequest) -> str:
        """Собирает дайджест и форматирует в Markdown."""
        response = self.build(request)
        if not response.items:
            return "Нет новостей по заданным критериям."
        
        llm_json = self.generator.generate(response)
        return format_digest_markdown(llm_json, response.items)


builder = DigestBuilder()