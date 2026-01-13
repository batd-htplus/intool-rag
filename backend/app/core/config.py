import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "RAG Backend")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    RAG_SERVICE_URL: str = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")

    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


settings = Settings()
