import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

    CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    CF_MODEL_ID = "@cf/baai/bge-m3"

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = "knu_info_2026"
    

    
    # Email Config (Mock)
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

settings = Settings()
