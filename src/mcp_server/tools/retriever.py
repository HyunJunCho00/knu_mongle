import os
import time
import mmh3
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter
from qdrant_client import QdrantClient, models
from kiwipiepy import Kiwi
from ..cloudflare_client import cf_client

class HybridRetriever:
    """
    하이브리드 검색 시스템
    - Dense: Cloudflare BGE-M3 임베딩
    - Sparse: Kiwi 형태소 분석 + MMH3 해싱
    - Qdrant을 이용한 벡터/그래프 검색
    """
    
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30
        )
        self.collection_name = os.getenv("COLLECTION_NAME", "campus_docs")
        self.kiwi = Kiwi(model_type='sbg')
        self.stop_tags = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF'}
    
    async def _encode_dense(self, text: str) -> List[float]:
        """Cloudflare BGE-M3 임베딩 생성"""
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json={"text": [text]},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                return result["result"]["data"][0]
        except Exception as e:
            print(f"Dense encoding error: {e}")
            return [0.0] * 1024
    
    def _encode_sparse(self, text: str) -> Tuple[Optional[List[int]], Optional[List[float]]]:
        """Kiwi 형태소 분석을 이용한 sparse 벡터 생성"""
        tokens = self.kiwi.tokenize(text)
        keywords = [
            t.form for t in tokens
            if t.tag not in self.stop_tags and len(t.form) > 1 and not t.form.isdigit()
        ]
        
        if not keywords:
            return None, None
        
        term_counts = Counter(keywords)
        indices = []
        values = []
        
        for term, count in term_counts.items():
            idx = mmh3.hash(term, signed=False)
            val = float(np.sqrt(count))
            indices.append(idx)
            values.append(val)
        
        return indices, values
    
    async def search(
        self,
        query: str,
        department: str = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        하이브리드 검색 실행
        
        Args:
            query: 검색 쿼리
            department: 부서 필터 (선택적)
            limit: 반환할 결과 수
        
        Returns:
            Dict: 검색 결과 및 메타데이터
        """
        start_time = time.time()
        
        # Dense 및 Sparse 벡터 병렬 생성
        dense_task = self._encode_dense(query)
        sparse_result = self._encode_sparse(query)
        
        dense_vec = await dense_task
        sp_indices, sp_values = sparse_result
        
        # 필터 구성
        search_filter = None
        if department and department != "공통":
            search_filter = models.Filter(
                must=[models.FieldCondition(
                    key="dept",
                    match=models.MatchValue(value=department)
                )]
            )
        
        # Qdrant 검색 쿼리 구성
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=50,
                filter=search_filter
            )
        ]
        
        if sp_indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(indices=sp_indices, values=sp_values),
                using="sparse",
                limit=50,
                filter=search_filter
            ))
        
        # Qdrant 검색 실행
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        
        # 결과 파싱
        parsed_results = []
        for point in results.points:
            payload = point.payload or {}
            parsed_results.append({
                "score": float(point.score),
                "title": payload.get("title", ""),
                "content": payload.get("content", ""),
                "url": payload.get("url", ""),
                "dept": payload.get("dept", ""),
                "date": payload.get("date", "")
            })
        
        elapsed = time.time() - start_time
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(parsed_results),
            "processing_time_ms": int(elapsed * 1000),
            "results": parsed_results
        }

# MCP 도구용 인터페이스
retriever = HybridRetriever()

async def campus_search_tool(query: str, department: str = None, limit: int = 5) -> str:
    """캠퍼스 문서 검색 MCP 도구"""
    result = await retriever.search(query, department, limit)
    return json.dumps(result, ensure_ascii=False)