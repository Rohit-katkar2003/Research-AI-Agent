"""
core/config.py — Centralised settings via pydantic-settings
All env vars loaded from .env file automatically
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
import os 

class Settings(BaseSettings):
   
    APP_NAME: str = "ResearchAI Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    DEFAULT_MODEL: str = "0.6b"
    MODEL_0_6B: str = "Rohit-Katkar2003/qwen-0.6b-fine-tune-4"
    MODEL_1_5B: str = "Rohit-Katkar2003/qwen-1.5b-fine-tune"   
    MAX_SEQ_LENGTH: int = 2048
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9

    # ── Tavily ──────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TRAVILY_API_KEY")
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "advanced"   # basic | advanced

    # ── Agent ───────────────────────────────────────
    MAX_SEARCH_ITERATIONS: int = 3
    MAX_SOURCES_PER_QUERY: int = 5

    # ── Paths ───────────────────────────────────────
    REPORTS_DIR: str = "./reports"
    CACHE_DIR: str = "./cache"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()