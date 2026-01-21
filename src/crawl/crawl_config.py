import os
import re
import json
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
NOTICES_DIR = DATA_DIR / "notices"
IMAGE_DIR = DATA_DIR / "images"
SCHEDULES_DIR = DATA_DIR / "schedules"
CURRICULUM_DIR = DATA_DIR / "curriculum"

for directory in [DATA_DIR, NOTICES_DIR, IMAGE_DIR, SCHEDULES_DIR, CURRICULUM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"[Config] Project Root: {PROJECT_ROOT}")
print(f"[Config] Data Dir: {DATA_DIR}")
print(f"[Config] Notices Dir: {NOTICES_DIR}")

CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "notices_dir": str(NOTICES_DIR),
    "image_dir": str(IMAGE_DIR),
    "schedules_dir": str(SCHEDULES_DIR),
    "curriculum_dir": str(CURRICULUM_DIR),
    
    "cutoff_date": "2025-01-01",
    "max_workers": 10,
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive"
    }
}

def clean_text(text: str) -> str:
    """텍스트 정제"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def get_last_crawled_date(file_name: str) -> str:
    """저장된 파일에서 마지막 크롤링 날짜 확인"""
    file_path = Path(CONFIG["notices_dir"]) / file_name
    
    last_date = CONFIG["cutoff_date"]
    
    if not file_path.exists():
        return last_date

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    if data.get('date') and data['date'] > last_date:
                        last_date = data['date']
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[Warning] Failed to read last date from {file_name}: {e}")
        
    return last_date