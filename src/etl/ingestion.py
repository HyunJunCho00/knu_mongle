import glob
import json
import os
import time
from typing import List, Dict, Set
from qdrant_client import models
from src.core.config import settings
from src.core.database import get_client, init_collection
from src.etl.encoders import CloudflareDenseEncoder, ProductionSparseEncoder
from src.etl.utils import (
    load_state, 
    save_state, 
    generate_doc_hash, 
    generate_chunk_id, 
    chunk_text
)

def main():
    print("=== Starting Ingestion Process (Deterministic IDs) ===")
    
    # 1. 리소스 초기화
    client = get_client()
    init_collection(client)
    
    dense_encoder = CloudflareDenseEncoder()
    sparse_encoder = ProductionSparseEncoder()
    
    # 이미 처리완료된 문서 리스트 로드
    processed_hashes = load_state()
    print(f"Loaded {len(processed_hashes)} processed docs from state.")

    # 2. 파일 목록 로드
    jsonl_files = glob.glob(os.path.join(settings.DATA_DIR, "**/*.jsonl"), recursive=True)
    
    buffer_points = []
    buffer_doc_hashes = set()
    
    # Cloudflare API 제한을 고려한 배치 사이즈
    BATCH_SIZE = settings.BATCH_SIZE  # e.g., 10 ~ 30
    
    total_chunks_ingested = 0

    for file_path in jsonl_files:
        # 마스터 파일 제외
        if "knu_master.jsonl" in file_path: continue
        
        filename = os.path.basename(file_path)
        dept_guess = filename.split("_")[0]
        
        if not os.path.exists(file_path): continue
        
        print(f"Reading file: {filename}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: continue
                    
                    doc = json.loads(line)
                    doc_hash = generate_doc_hash(doc)

                    # [Check 1] 이미 처리된 문서라면 아예 건너뜀 (속도 최적화)
                    if doc_hash in processed_hashes:
                        continue
                    
                    # [Step 1] 문서 파싱 및 청킹
                    title = doc.get('title', '')
                    content = doc.get('content', '') or title
                    url = doc.get('url', '')
                    date = doc.get('date', '')
                    dept = doc.get('dept', dept_guess)
                    
                    full_text = f"[{dept}] {title}\n\n{content}"
                    chunks = chunk_text(full_text, settings.CHUNK_SIZE)
                    
                    # [Step 2] Point 생성 (아직 업로드 안함)
                    for i, chunk in enumerate(chunks):
                        # 너무 짧은 청크 무시
                        if len(chunk) < 5: continue
                        
                        # [중요] ID 생성 시 고정된 규칙 사용 (재실행 시 중복 방지)
                        point_id = generate_chunk_id(doc_hash, i)
                        
                        # Sparse Vector 생성 (CPU 작업)
                        sp_indices, sp_values = sparse_encoder.encode(chunk, title, dept)
                        if not sp_indices:
                            sp_indices, sp_values = sparse_encoder.get_fallback_vector(chunk)
                            
                        # 버퍼에 담기 (Dense는 나중에 한꺼번에)
                        buffer_points.append({
                            "id": point_id,
                            "text": chunk,
                            "sparse_indices": sp_indices,
                            "sparse_values": sp_values,
                            "payload": {
                                "title": title,
                                "dept": dept,
                                "url": url,
                                "date": date,
                                "content": chunk,
                                "doc_id": doc_hash,
                                "chunk_idx": i
                            }
                        })
                    
                    # 현재 처리 중인 문서 해시 기록
                    buffer_doc_hashes.add(doc_hash)
                    
                    # [Step 3] 버퍼가 찼으면 업로드 실행
                    if len(buffer_points) >= BATCH_SIZE:
                        _upload_batch(client, dense_encoder, buffer_points)
                        
                        # 성공 시 상태 저장
                        processed_hashes.update(buffer_doc_hashes)
                        save_state(processed_hashes)
                        
                        total_chunks_ingested += len(buffer_points)
                        print(f"Uploaded {len(buffer_points)} chunks. (Total: {total_chunks_ingested})")
                        
                        # 버퍼 초기화
                        buffer_points = []
                        buffer_doc_hashes = set()
                        
                        # API Rate Limit 보호
                        time.sleep(0.5)

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing doc: {e}")
                    # 에러가 나도 다음 문서 계속 진행

    # [Step 4] 남은 버퍼 처리 (마지막 배치)
    if buffer_points:
        _upload_batch(client, dense_encoder, buffer_points)
        processed_hashes.update(buffer_doc_hashes)
        save_state(processed_hashes)
        total_chunks_ingested += len(buffer_points)
        print(f"Uploaded final batch: {len(buffer_points)} chunks.")

    print("=== Ingestion Complete ===")

def _upload_batch(client, dense_encoder, points_data):
    """
    실제 Dense Encoding 및 Upsert 수행 함수
    """
    if not points_data: return

    texts = [p['text'] for p in points_data]
    
    # Dense Encoding (Cloudflare API)
    try:
        dense_vectors = dense_encoder.encode(texts)
    except Exception as e:
        print(f"[Fatal] Dense encoding failed: {e}")
        return # 실패 시 이번 배치는 스킵 (다음 실행 때 다시 시도됨)

    if not dense_vectors:
        return

    qdrant_points = []
    for j, item in enumerate(points_data):
        if j >= len(dense_vectors): break
        
        qdrant_points.append(models.PointStruct(
            id=item['id'],
            vector={
                "dense": dense_vectors[j],
                "sparse": models.SparseVector(
                    indices=item['sparse_indices'], 
                    values=item['sparse_values']
                )
            },
            payload=item['payload']
        ))
    
    try:
        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=qdrant_points
        )
    except Exception as e:
        print(f"[Fatal] Qdrant upsert failed: {e}")