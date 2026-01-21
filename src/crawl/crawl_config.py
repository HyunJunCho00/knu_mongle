import os
import re
import json
from pathlib import Path

# =========================================================
# 1. 경로 설정 (Path Configuration)
# =========================================================

# 현재 파일(config.py)의 절대 경로를 구하기
CURRENT_FILE = Path(__file__).resolve()

# 프로젝트 루트(Project Root) 찾기
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"

# 디렉토리가 없으면 자동으로 생성 (부모 폴더 포함)
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Config] Project Root: {PROJECT_ROOT}")
print(f"[Config] Data Dir:     {DATA_DIR}")

# =========================================================
# 2. 설정 값 (Global Configuration)
# =========================================================

CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "image_dir": str(IMAGE_DIR),
    
    "cutoff_date": "2025-01-01",
    "max_workers": 10,
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive"
    }
}

# =========================================================
# 3. 유틸리티 함수 (Utility Functions)
# =========================================================

def clean_text(text: str) -> str:
    """
    텍스트의 불필요한 공백과 줄바꿈을 제거하여 정제합니다.
    """
    if not text:
        return ""
    # 연속된 공백, 탭, 줄바꿈을 단일 공백으로 치환
    return re.sub(r'\s+', ' ', str(text)).strip()

def get_last_crawled_date(file_name: str) -> str:
    """
    저장된 파일(jsonl)을 읽어 마지막으로 수집된 날짜를 확인합니다.
    file_name: data_dir 내부의 파일명 (예: 'knu_notice.jsonl')
    """
    # 설정된 data_dir 경로와 파일명을 결합
    file_path = Path(CONFIG["data_dir"]) / file_name
    
    last_date = CONFIG["cutoff_date"]
    
    if not file_path.exists():
        return last_date

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: continue
                    
                    data = json.loads(line)
                    # date 필드가 있고, 현재 last_date보다 최신이면 갱신
                    if data.get('date') and data['date'] > last_date:
                        last_date = data['date']
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[Warning] Failed to read last date from {file_name}: {e}")
        
    return last_date
