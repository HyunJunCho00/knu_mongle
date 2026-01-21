import requests
import re
import os
import numpy as np
import mmh3
from kiwipiepy import Kiwi
from typing import List, Tuple, Optional, Dict
from src.core.config import settings

class CloudflareDenseEncoder:
    """Cloudflare Workers AI REST API를 사용하는 Dense 인코더"""
    
    def __init__(self):
        self.api_url = f"https://api.cloudflare.com/client/v4/accounts/{settings.CF_ACCOUNT_ID}/ai/run/{settings.CF_MODEL_ID}"
        self.headers = {"Authorization": f"Bearer {settings.CF_API_TOKEN}"}
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        try:
            # Cloudflare BGE-M3 API 호출
            payload = {"text": texts}
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result.get("success"):
                # 결과 포맷이 {'result': {'data': [[...], [...]]}} 형태라고 가정
                return result["result"]["data"]
            else:
                print(f"[Error] Cloudflare API Error: {result.get('errors')}")
                return []
                
        except Exception as e:
            print(f"[Error] Dense encoding failed: {str(e)}")
            return []

class ProductionSparseEncoder:
    """학사 문서에 최적화된 Production-grade Sparse 인코더"""
    
    def __init__(self):
        # GitHub Actions 환경(CPU)에서도 가볍게 동작하는 Kiwi 설정
        self.kiwi = Kiwi(num_workers=0, model_type='sbg')
        
        self.stop_tags = {
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
            'EP', 'EF', 'EC', 'ETN', 'ETM', 'SP', 'SS', 'SE', 'SO', 'SL', 
            'SH', 'SN', 'SF', 'SY', 'IC', 'XPN', 'XSN', 'XSV', 'XSA', 
            'XR', 'MM', 'MAG', 'MAJ', 'VCP', 'VCN', 'VA', 'VV', 'VX'
        }
        
        # 학사 관련 키워드 가중치 사전
        self.academic_keywords = {
            '졸업': 3.0, '졸업요건': 3.0, '수강신청': 3.0, '장학금': 3.0, 
            '전과': 3.0, '복수전공': 3.0, '부전공': 3.0, '계절학기': 3.0,
            '학점': 2.5, '이수': 2.5, '전공필수': 2.5, '교양': 2.5,
            '인턴십': 2.0, '취업': 2.0, '모집': 1.5, '신청': 1.5
        }

    def preprocess_text(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)  # HTML 제거
        text = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣.,!?;:()\'\"~\-/\d]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def encode(self, text: str, title: str = "", dept: str = "") -> Tuple[Optional[List[int]], Optional[List[float]]]:
        try:
            if not text or len(text.strip()) < 5:
                return None, None
            
            # 텍스트 전처리 및 부서명 결합 (Context 강화)
            processed_text = self.preprocess_text(text)
            full_text = f"{dept} {title} {processed_text}"
            
            tokens = self.kiwi.tokenize(full_text)
            
            term_weights = {}
            for t in tokens:
                form = t.form
                if t.tag in self.stop_tags or len(form) < 2:
                    continue
                
                # 기본 가중치
                weight = 1.0
                
                # 사전 정의된 키워드 가중치 적용
                if form in self.academic_keywords:
                    weight = self.academic_keywords[form]
                elif form in title:
                    weight = 2.0  # 제목에 포함된 단어 가중치
                elif form in dept:
                    weight = 2.5  # 학과명 가중치
                
                term_weights[form] = term_weights.get(form, 0) + weight

            if not term_weights:
                return None, None

            # Sparse Vector 생성
            indices = []
            values = []
            
            for term, weight in term_weights.items():
                idx = mmh3.hash(term, signed=False)
                # BM25 유사 가중치 (간소화됨)
                final_val = float(np.sqrt(weight))
                indices.append(idx)
                values.append(final_val)
                
            return indices, values

        except Exception as e:
            print(f"[Error] Sparse encoding: {str(e)}")
            return None, None

    def get_fallback_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """인코딩 실패 시 사용할 기본 벡터"""
        fallback_terms = ["공지", "학교", "안내"]
        indices = [mmh3.hash(t, signed=False) for t in fallback_terms]
        values = [1.0] * len(indices)
        return indices, values