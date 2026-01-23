import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "RAG Backend")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    RAG_SERVICE_URL: str = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")


settings = Settings()
