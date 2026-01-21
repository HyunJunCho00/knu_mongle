import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

    CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "knu_info_2026"
    
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    NOTICES_DIR = DATA_DIR / "notices"
    IMAGES_DIR = DATA_DIR / "images"
    SCHEDULES_DIR = DATA_DIR / "schedules"
    CURRICULUM_DIR = DATA_DIR / "curriculum"
    
    STATE_FILE = DATA_DIR / "ingestion_state.json"
    
    CHUNK_SIZE = 900
    BATCH_SIZE = 10

settings = Settings()

for directory in [
    settings.DATA_DIR,
    settings.NOTICES_DIR,
    settings.IMAGES_DIR,
    settings.SCHEDULES_DIR,
    settings.CURRICULUM_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
