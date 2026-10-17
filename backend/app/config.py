"""Configuration settings for Transformer Explorer.
By Hammad, AI Engineer.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Locate and load .env file from backend or root directory
BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent

if (BACKEND_DIR / ".env").exists():
    load_dotenv(BACKEND_DIR / ".env")
elif (ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env")
else:
    load_dotenv()  # Fallback to default search

# Allowed models list - strictly enforces safety against untrusted remote code
ALLOWED_MODELS: List[str] = ["distilgpt2", "gpt2"]

class Settings:
    PROJECT_NAME: str = "Real Transformer Internals Explorer"
    AUTHOR: str = "By Hammad, AI Engineer"
    VERSION: str = "1.0.0"

    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "distilgpt2")
    ALLOWED_MODELS: List[str] = ALLOWED_MODELS
    ATTN_IMPLEMENTATION: str = "eager"  # Required for extracting real attention weights
    TRUST_REMOTE_CODE: bool = False     # Strict security requirement
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "512"))

    # Device selection: 'auto', 'cpu', 'cuda'
    DEVICE_CONFIG: str = os.getenv("DEVICE", "auto")

    # Server settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # CORS origins
    _cors_raw: str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    CORS_ORIGINS: List[str] = [origin.strip() for origin in _cors_raw.split(",") if origin.strip()]

    # Gemini settings (Phase 6, optional - app runs fully without key)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

settings = Settings()
