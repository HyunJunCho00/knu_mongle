import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Vertex AI / Gemini Configuration
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    VERTEX_AI_MODEL = "gemini-3.0-flash" 
    
    # API Keys
    KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
    
    # Cloudflare for Embeddings
    CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    
    # Vector DB
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "knu_info"
    
    # Email Configuration
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    
    # File Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    NOTICES_DIR = DATA_DIR / "notices"
    SCHEDULES_DIR = DATA_DIR / "schedules"
    CURRICULUM_DIR = DATA_DIR / "curriculum"
    TEMPLATES_DIR = PROJECT_ROOT / "templates"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    ATTACHMENTS_DIR=DATA_DIR / "attachments"

    STATE_FILE = DATA_DIR / "ingestion_state.json"
    
    # Search Configuration
    CHUNK_SIZE = 900
    BATCH_SIZE = 10


settings = Settings()

# Create necessary directories
for directory in [
    settings.DATA_DIR,
    settings.NOTICES_DIR,
    settings.SCHEDULES_DIR,
    settings.CURRICULUM_DIR,
    settings.TEMPLATES_DIR,
    settings.ATTACHMENTS_DIR,
    settings.OUTPUT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)