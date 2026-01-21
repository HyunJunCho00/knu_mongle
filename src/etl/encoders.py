import requests
import re
import numpy as np
import mmh3
from kiwipiepy import Kiwi
from typing import List, Tuple, Optional
from src.core.config import settings

class CloudflareDenseEncoder:
    """Cloudflare Workers AI BGE-M3 Dense Encoder"""
    
    def __init__(self):
        self.api_url = f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/baai/bge-m3"
        self.headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts using Cloudflare BGE-M3 API
        Returns: List of 1024-dimensional dense vectors
        """
        try:
            payload = {"text": texts}
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("success"):
                embeddings = result["result"]["data"]
                
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    if isinstance(embeddings[0], list):
                        return embeddings
                    else:
                        return [embeddings]
                else:
                    print(f"Unexpected embedding format: {result}")
                    return [[0.0] * 1024 for _ in texts]
            else:
                print(f"Cloudflare API Error: {result.get('errors')}")
                return [[0.0] * 1024 for _ in texts]
                
        except Exception as e:
            print(f"Dense encoding failed: {str(e)}")
            return [[0.0] * 1024 for _ in texts]
    
    def encode_single(self, text: str) -> List[float]:
        """Encode a single text"""
        result = self.encode([text])
        return result[0] if result else [0.0] * 1024

class ProductionSparseEncoder:
    """
    BM25-inspired sparse encoder using Kiwi morphological analysis
    """
    
    def __init__(self):
        self.kiwi = Kiwi(num_workers=0, model_type='sbg')
        
        self.stop_tags = {
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
            'EP', 'EF', 'EC', 'ETN', 'ETM', 'SP', 'SS', 'SE', 'SO', 'SL',
            'SH', 'SN', 'SF', 'SY', 'IC', 'XPN', 'XSN', 'XSV', 'XSA',
            'XR', 'MM', 'MAG', 'MAJ', 'VCP', 'VCN', 'VA', 'VV', 'VX'
        }
        
        self.academic_keywords = {
            '졸업': 3.0, '졸업요건': 3.0, '수강신청': 3.0, '장학금': 3.0,
            '전과': 3.0, '복수전공': 3.0, '부전공': 3.0, '계절학기': 3.0,
            '학점': 2.5, '이수': 2.5, '전공필수': 2.5, '교양': 2.5,
            '인턴십': 2.0, '취업': 2.0, '모집': 1.5, '신청': 1.5,
            '비자': 3.0, 'VISA': 3.0, '외국인': 2.5, '유학생': 2.5,
            '체류': 2.5, '등록': 2.0, '입학': 2.0,'비자':3.0,'인턴':3.0
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣.,!?;:()\'\"~\-/\d]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def encode(self, text: str, title: str = "", dept: str = "") -> Tuple[Optional[List[int]], Optional[List[float]]]:
        """
        Encode text into sparse vector using morphological analysis
        
        Returns:
            indices: List of hash indices
            values: List of term weights
        """
        try:
            if not text or len(text.strip()) < 5:
                return None, None
            
            processed_text = self.preprocess_text(text)
            full_text = f"{dept} {title} {processed_text}"
            
            tokens = self.kiwi.tokenize(full_text)
            
            term_weights = {}
            for t in tokens:
                form = t.form
                if t.tag in self.stop_tags or len(form) < 2:
                    continue
                
                weight = 1.0
                
                if form in self.academic_keywords:
                    weight = self.academic_keywords[form]
                elif form in title:
                    weight = 2.0
                elif form in dept:
                    weight = 2.5
                
                term_weights[form] = term_weights.get(form, 0) + weight

            if not term_weights:
                return None, None

            indices = []
            values = []
            
            for term, weight in term_weights.items():
                idx = mmh3.hash(term, signed=False)
                final_val = float(np.sqrt(weight))
                indices.append(idx)
                values.append(final_val)
                
            return indices, values

        except Exception as e:
            print(f"Sparse encoding error: {str(e)}")
            return None, None

    def get_fallback_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """Fallback sparse vector for error cases"""
        fallback_terms = ["공지", "학교", "안내"]
        indices = [mmh3.hash(t, signed=False) for t in fallback_terms]
        values = [1.0] * len(indices)
        return indices, values