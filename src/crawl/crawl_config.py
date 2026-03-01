import os
import re
import json
from datetime import date, timedelta
from pathlib import Path
try:
    from core.config import Settings
except ImportError:
    from src.core.config import Settings

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

DATA_DIR = Settings.DATA_DIR
NOTICES_DIR = Settings.NOTICES_DIR
SCHEDULES_DIR = Settings.SCHEDULES_DIR
CURRICULUM_DIR = Settings.CURRICULUM_DIR
ATTACHMENTS_DIR=Settings.ATTACHMENTS_DIR

# Crawl policy (fixed constants)
COLD_START = True
COLD_START_DATE = "2026-01-01"
CUTOFF_DATE = COLD_START_DATE if COLD_START else (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")

CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "notices_dir": str(NOTICES_DIR),
    "schedules_dir": str(SCHEDULES_DIR),
    "curriculum_dir": str(CURRICULUM_DIR),
    "attachments_dir": str(ATTACHMENTS_DIR),  
    
    "cold_start": COLD_START,
    "cold_start_date": COLD_START_DATE,
    "cutoff_date": CUTOFF_DATE,
    "max_workers": 20,
    "max_file_workers": 6,
    "max_image_workers": 6,
    "request_delay": 0.02,
    "extract_text_exts": [".pdf", ".docx", ".hwp", ".hwpx", ".xlsx", ".xls", ".pptx", ".txt", ".csv"],
    "download_file_exts": [".pdf", ".docx", ".hwp", ".hwpx", ".xlsx", ".xls", ".pptx"],
    "image_exts": [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"],
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
