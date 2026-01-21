import json
import os
import hashlib
import uuid
from typing import Set, List
from kiwipiepy import Kiwi
from src.core.config import settings

# Kiwi 초기화 (전역 로드)
kiwi = Kiwi(model_type='sbg')

def load_state() -> Set[str]:
    """이미 처리된 문서의 해시값 로드"""
    if os.path.exists(settings.STATE_FILE):
        try:
            with open(settings.STATE_FILE, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_state(hashes: Set[str]):
    """처리된 문서 해시값 저장 (Atomic)"""
    temp_file = settings.STATE_FILE + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(list(hashes), f)
    os.replace(temp_file, settings.STATE_FILE)

def generate_doc_hash(doc: dict) -> str:
    """
    문서 고유 식별자 생성
    (URL + Date + Title) 조합이 같으면 동일 문서로 취급
    """
    unique_str = f"{doc.get('url', '')}_{doc.get('date', '')}_{doc.get('title', '')}"
    return hashlib.sha256(unique_str.encode()).hexdigest()

def generate_chunk_id(doc_hash: str, chunk_idx: int) -> str:
    """
    [ID 고정] 문서 해시와 청크 순서를 조합하여 항상 동일한 UUID 생성 (Idempotency 보장)
    크롤러가 중복된 내용을 가져와도 Qdrant에서 ID가 같으므로 덮어쓰기(Upsert)됨.
    """
    seed = f"{doc_hash}_{chunk_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

def chunk_text(text: str, size: int = 900) -> List[str]:
    """
    Kiwi를 활용한 문장 단위 청킹
    """
    if not text: return []
    
    try:
        sentences = kiwi.split_into_sents(text)
        sents_text = [s.text for s in sentences]
    except:
        return [text[i:i+size] for i in range(0, len(text), size)]

    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sents_text:
        sent_len = len(sent)
        if current_len + sent_len <= size:
            current_chunk.append(sent)
            current_len += sent_len
        else:
            if current_chunk: chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
            
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks