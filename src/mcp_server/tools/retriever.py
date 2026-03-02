import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client import QdrantClient, models
from src.core.config import settings


class HybridRetriever:
    """
    Hybrid retriever with:
    - Query decomposition
    - Hybrid dense+sparse retrieval
    - Rule-based reranking
    - Query-time context packing (neighbor nodes)
    """

    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
            cloud_inference=True,
        )
        self.collection_name = os.getenv("COLLECTION_NAME", settings.COLLECTION_NAME)

        self.account_id = os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = os.getenv("CF_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN")
        self.dense_model = "@cf/baai/bge-m3"
        self.node_type_weight = {
            "section": 1.12,
            "table_row": 1.10,
            "list_item": 1.06,
            "paragraph": 1.0,
        }

    async def _encode_dense(self, text: str) -> List[float]:
        if not self.account_id or not self.api_token:
            return [0.0] * 1024

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.dense_model}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        try:
            async with httpx.AsyncClient(timeout=12) as client:
                response = await client.post(url, headers=headers, json={"text": [text]})
                response.raise_for_status()
                result = response.json()
                vectors = result.get("result", {}).get("data", [])
                if vectors and isinstance(vectors[0], list):
                    return vectors[0]
                if isinstance(vectors, list) and len(vectors) == 1024:
                    return vectors
        except Exception:
            pass
        return [0.0] * 1024

    @staticmethod
    def _build_sparse_query(text: str) -> models.Document:
        return models.Document(text=text, model="qdrant/bm25")

    @staticmethod
    def _decompose_query(query: str) -> List[str]:
        q = query.strip()
        if not q:
            return []
        splits = re.split(r"\?|,| 그리고 | 및 | 또는 |/|;", q)
        subs = [s.strip() for s in splits if len(s.strip()) >= 2]
        if not subs:
            return [q]
        # keep top 3 sub-queries + original
        out = subs[:3]
        if q not in out:
            out.append(q)
        return out

    @staticmethod
    def _tokenize_for_overlap(text: str) -> List[str]:
        return re.findall(r"[가-힣A-Za-z0-9]{2,}", (text or "").lower())

    def _rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        q_tokens = set(self._tokenize_for_overlap(query))
        reranked = []
        for item in items:
            payload = item.get("payload", {})
            content = payload.get("content", "")
            title = payload.get("title", "")
            combined = f"{title} {content}"
            d_tokens = set(self._tokenize_for_overlap(combined))
            overlap = len(q_tokens.intersection(d_tokens))
            base = float(item.get("score", 0.0))
            node_type = payload.get("node_type", payload.get("chunk_type", "paragraph"))
            nt_w = self.node_type_weight.get(node_type, 1.0)
            rerank_score = (base * nt_w) + (0.015 * overlap)
            item["rerank_score"] = rerank_score
            reranked.append(item)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked

    def _pack_neighbors(self, item: Dict[str, Any]) -> Dict[str, Any]:
        payload = item.get("payload", {})
        doc_url = payload.get("url")
        idx = payload.get("chunk_index")
        if doc_url is None or idx is None:
            return item

        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            return item

        must = [
            models.FieldCondition(key="url", match=models.MatchValue(value=doc_url)),
            models.FieldCondition(
                key="chunk_index",
                range=models.Range(gte=max(idx_int - 1, 0), lte=idx_int + 1),
            ),
        ]
        filt = models.Filter(must=must)
        neighbors, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filt,
            with_payload=True,
            with_vectors=False,
            limit=5,
        )
        neighbors_sorted = sorted(
            neighbors,
            key=lambda p: int((p.payload or {}).get("chunk_index", 0)),
        )
        context = "\n\n".join([(p.payload or {}).get("content", "") for p in neighbors_sorted if (p.payload or {}).get("content")])
        if context:
            item["packed_context"] = context
        return item

    async def _search_once(self, query: str, department: Optional[str], limit: int) -> List[Dict[str, Any]]:
        dense_task = self._encode_dense(query)
        sparse_doc = self._build_sparse_query(query)
        dense_vec = await dense_task

        search_filter = None
        if department and department != "공통":
            search_filter = models.Filter(
                must=[models.FieldCondition(key="dept", match=models.MatchValue(value=department))]
            )

        prefetch = [
            models.Prefetch(query=dense_vec, using="dense", limit=max(50, limit * 6), filter=search_filter)
        ]
        prefetch.append(
            models.Prefetch(
                query=sparse_doc,
                using="sparse",
                limit=max(50, limit * 6),
                filter=search_filter,
            )
        )

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=max(30, limit * 4),
            with_payload=True,
            with_vectors=False,
        )

        out = []
        for p in results.points:
            out.append({"id": p.id, "score": float(p.score), "payload": p.payload or {}})
        return out

    async def search(
        self,
        query: str,
        department: str = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        start_time = time.time()
        sub_queries = self._decompose_query(query)
        if not sub_queries:
            return {
                "status": "success",
                "query": query,
                "sub_queries": [],
                "results_count": 0,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "results": [],
            }

        merged: Dict[str, Dict[str, Any]] = {}
        for sq in sub_queries:
            one = await self._search_once(sq, department=department, limit=limit)
            for item in one:
                key = str(item.get("id"))
                if key not in merged:
                    merged[key] = item
                else:
                    # keep best original hybrid score
                    merged[key]["score"] = max(float(merged[key]["score"]), float(item["score"]))

        reranked = self._rerank(query, list(merged.values()))
        selected = reranked[: max(limit * 2, 6)]
        packed = [self._pack_neighbors(it) for it in selected]
        final = packed[:limit]

        parsed_results = []
        for it in final:
            payload = it.get("payload", {})
            parsed_results.append(
                {
                    "score": float(it.get("rerank_score", it.get("score", 0.0))),
                    "title": payload.get("title", ""),
                    "content": payload.get("content", ""),
                    "packed_context": it.get("packed_context", payload.get("content", "")),
                    "url": payload.get("url", ""),
                    "dept": payload.get("dept", ""),
                    "date": payload.get("date", ""),
                    "node_type": payload.get("node_type", payload.get("chunk_type", "")),
                    "node_path": payload.get("node_path", ""),
                    "section_header": payload.get("section_header", ""),
                    "evidence": {
                        "point_id": str(it.get("id", "")),
                        "source_url": payload.get("url", ""),
                        "section_path": payload.get("node_path", ""),
                        "chunk_index": payload.get("chunk_index", -1),
                    },
                }
            )

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "query": query,
            "sub_queries": sub_queries,
            "results_count": len(parsed_results),
            "processing_time_ms": int(elapsed * 1000),
            "results": parsed_results,
        }


retriever = HybridRetriever()


async def campus_search_tool(query: str, department: str = None, limit: int = 5) -> str:
    result = await retriever.search(query, department, limit)
    return json.dumps(result, ensure_ascii=False)
