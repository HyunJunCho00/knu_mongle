import time
import requests
import mmh3
import numpy as np
from typing import List, Optional, Tuple
from collections import Counter
from qdrant_client import QdrantClient, models
from kiwipiepy import Kiwi
from src.core.config import settings  

class KNUSearcher:
    """
    Hybrid Searcher: Cloudflare Workers AI (Dense) + Kiwi (Sparse)
    """
    def __init__(self):
        # 1. Qdrant Client 초기화
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,
            https=True,
            timeout=60,
            verify=False
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # 2. Sparse Encoder Setup (Kiwi) - embedding.txt와 동일한 로직 유지를 위해 로컬 실행
        print("[System] Initializing Kiwi for Sparse Encoding...")
        self.kiwi = Kiwi()
        
        # [cite_start]embedding.txt의 불용어 태그 리스트와 동일하게 맞춤 [cite: 8]
        self.stop_tags = {
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
            'EP', 'EF', 'EC', 'ETN', 'ETM', 'SP', 'SS', 'SE', 'SO', 'SL', 
            'SH', 'SN', 'SF', 'SY', 'IC', 'XPN', 'XSN', 'XSV', 'XSA', 
            'XR', 'MM', 'MAG', 'MAJ', 'VCP', 'VCN', 'VA', 'VV', 'VX'
        }

    def _encode_dense(self, text: str) -> List[float]:
        """
        Cloudflare Workers AI (BGE-M3) API 호출
        """
        account_id = settings.CLOUDFLARE_ACCOUNT_ID
        api_token = settings.CLOUDFLARE_API_TOKEN
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        try:
            response = requests.post(url, headers=headers, json={"text": text}, timeout=5)
            result = response.json()
            
            # Cloudflare 응답 파싱 (성공 여부 확인)
            if result.get("success"):
                # 보통 result['result']['data'][0]에 벡터가 있음
                return result['result']['data'][0]
            else:
                print(f"[Error] Cloudflare API Error: {result}")
                return [0.0] * 1024  # 에러 시 0 벡터 반환 (검색 실패 방지)
                
        except Exception as e:
            print(f"[Error] Dense encoding failed: {e}")
            return [0.0] * 1024

    def _encode_sparse(self, text: str) -> Tuple[Optional[List[int]], Optional[List[float]]]:
        """
        [cite_start]Kiwi 형태소 분석 + MMH3 해싱 (embedding.txt 로직과 100% 일치시켜야 함) [cite: 30-45]
        """
        try:
            tokens = self.kiwi.tokenize(text)
            keywords = [
                t.form for t in tokens 
                if t.tag not in self.stop_tags and len(t.form) > 1
            ]
            
            if not keywords: 
                return None, None
            
            term_counts = Counter(keywords)
            indices = []
            values = []
            
            for term, count in term_counts.items():
                # [cite_start]Qdrant Sparse Index용 해싱 [cite: 44]
                idx = mmh3.hash(term, signed=False)
                # Query-side weighting: SQRT(TF) 적용
                val = float(np.sqrt(count)) 
                
                indices.append(idx)
                values.append(val)
                
            return indices, values
        except Exception as e:
            print(f"[Warning] Sparse encoding error: {e}")
            return None, None

    def search(self, query: str, target_dept: str = None, final_k: int = 5):
        """
        Hybrid Search 수행 및 결과 파싱
        """
        start_time = time.perf_counter()
        
        # 1. 벡터 변환 (API + Local CPU)
        dense_vec = self._encode_dense(query)
        sp_indices, sp_values = self._encode_sparse(query)
        
        # 2. 필터 구성
        search_filter = None
        if target_dept and target_dept != "공통":
            search_filter = models.Filter(
                must=[models.FieldCondition(key="dept", match=models.MatchValue(value=target_dept))]
            )

        # 3. Prefetch 구성 (Dense + Sparse)
        prefetch = []
        prefetch.append(models.Prefetch(
            query=dense_vec, using="dense", limit=50, filter=search_filter
        ))
        
        if sp_indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(indices=sp_indices, values=sp_values),
                using="sparse", limit=50, filter=search_filter
            ))

        # 4. 검색 실행 (RRF)
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=final_k,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"[Error] Search failed: {e}")
            return []

        # 5. 결과 가공
        parsed_results = []
        if results and results.points:
            for point in results.points:
                payload = point.payload or {}
                parsed_results.append({
                    "score": point.score,
                    "title": payload.get("title", "제목 없음"),
                    "content": payload.get("content", ""),
                    "url": payload.get("url", ""),
                    "dept": payload.get("dept", "공통"),
                    "date": payload.get("date", "")
                })
        
        return parsed_results
