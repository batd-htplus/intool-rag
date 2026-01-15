"""
Configuration for model service
"""
import os
from pathlib import Path

class Config:
    """Model service configuration"""
    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    
    LLM_MODEL = os.getenv("LLM_MODEL", "Phi3:mini")
    LLM_DEVICE = os.getenv("LLM_DEVICE", "cpu")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
    
    MODELS_DIR = Path("/app/models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "2"))
    LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "1"))
    THREADPOOL_MAX_WORKERS = int(os.getenv("THREADPOOL_MAX_WORKERS", "4"))
    
    EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "45.0"))
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))
    ACQUIRE_TIMEOUT = float(os.getenv("ACQUIRE_TIMEOUT", "3.0"))
    OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "100.0"))

config = Config()

