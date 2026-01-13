import os
from pathlib import Path

class Config:
    """RAG service configuration"""
    
    MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:8002")
    
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE = 32
    
    # LLM
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LLM_DEVICE = os.getenv("LLM_DEVICE", "cpu")
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 1024
    LLM_RELEVANCE_THRESHOLD = 0.6
    
    # Vector Store
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = "documents"
    VECTOR_DIMENSION = 1024
    
    # Chunking
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    RETRIEVAL_TOP_K = 5
    
    EMBEDDING_QUERY_INSTRUCTION = os.getenv(
        "EMBEDDING_QUERY_INSTRUCTION",
        "Represent this sentence for searching relevant passages: "
    )
    EMBEDDING_PASSAGE_INSTRUCTION = os.getenv(
        "EMBEDDING_PASSAGE_INSTRUCTION",
        "" 
    )
    
    # Storage
    STORAGE_DIR = Path("/storage")
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()
