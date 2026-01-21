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
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,
            https=True,
            timeout=60,
            verify=False
        )
        self.collection_name = settings.COLLECTION_NAME

        print("[System] Initializing Kiwi for Sparse Encoding...")
        self.kiwi = Kiwi(num_workers=0, model_type='sbg')
        
        self.stop_tags = {
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
            'EP', 'EF', 'EC', 'ETN', 'ETM', 'SP', 'SS', 'SE', 'SO', 'SL',
            'SH', 'SN', 'SF', 'SY', 'IC', 'XPN', 'XSN', 'XSV', 'XSA',
            'XR', 'MM', 'MAG', 'MAJ', 'VCP', 'VCN', 'VA', 'VV', 'VX'
        }

    def _encode_dense(self, text: str) -> List[float]:
        """
        Cloudflare Workers AI (BGE-M3) API call
        """
        account_id = settings.CLOUDFLARE_ACCOUNT_ID
        api_token = settings.CLOUDFLARE_API_TOKEN
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"text": [text]},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                embeddings = result['result']['data']
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    if isinstance(embeddings[0], list):
                        return embeddings[0]
                    else:
                        return embeddings
                else:
                    print(f"[Error] Unexpected embedding format: {result}")
                    return [0.0] * 1024
            else:
                print(f"[Error] Cloudflare API Error: {result}")
                return [0.0] * 1024
                
        except Exception as e:
            print(f"[Error] Dense encoding failed: {e}")
            return [0.0] * 1024

    def _encode_sparse(self, text: str) -> Tuple[Optional[List[int]], Optional[List[float]]]:
        """
        Kiwi morphological analysis + MMH3 hashing
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
                idx = mmh3.hash(term, signed=False)
                val = float(np.sqrt(count))
                
                indices.append(idx)
                values.append(val)
                
            return indices, values
        except Exception as e:
            print(f"[Warning] Sparse encoding error: {e}")
            return None, None

    def search(self, query: str, target_dept: str = None, final_k: int = 5):
        """
        Hybrid Search with RRF fusion
        """
        start_time = time.perf_counter()
        
        dense_vec = self._encode_dense(query)
        sp_indices, sp_values = self._encode_sparse(query)
        
        search_filter = None
        if target_dept and target_dept != "공통":
            search_filter = models.Filter(
                must=[models.FieldCondition(
                    key="dept",
                    match=models.MatchValue(value=target_dept)
                )]
            )

        prefetch = []
        prefetch.append(models.Prefetch(
            query=dense_vec,
            using="dense",
            limit=50,
            filter=search_filter
        ))
        
        if sp_indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(indices=sp_indices, values=sp_values),
                using="sparse",
                limit=50,
                filter=search_filter
            ))

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

        parsed_results = []
        if results and results.points:
            for point in results.points:
                payload = point.payload or {}
                parsed_results.append({
                    "score": point.score,
                    "title": payload.get("title", "No title"),
                    "content": payload.get("content", ""),
                    "url": payload.get("url", ""),
                    "dept": payload.get("dept", "Common"),
                    "date": payload.get("date", "")
                })
        
        elapsed = time.perf_counter() - start_time
        print(f"[Search] Query: '{query}' | Results: {len(parsed_results)} | Time: {elapsed:.3f}s")
        
        return parsed_results
