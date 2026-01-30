import os
from pathlib import Path

class Config:
    """RAG service configuration"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB1rpq40ooQEWeQnihquZFbZg4NjJdf7vI")

    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    EMBEDDING_PROGRESSIVE_BATCH = int(os.getenv("EMBEDDING_PROGRESSIVE_BATCH", "20"))
    EMBEDDING_MAX_PARALLEL = int(os.getenv("EMBEDDING_MAX_PARALLEL", "3"))
    
    # LLM
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    LLM_DEVICE = os.getenv("LLM_DEVICE", "cpu")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
    LLM_RELEVANCE_THRESHOLD = float(os.getenv("LLM_RELEVANCE_THRESHOLD", "0.4"))
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "150.0"))
    
    # Reranker
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "10"))
    
    # Vector Dimension
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))
    
    # Chunking strategy
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "512"))
    CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "50"))
    SEMANTIC_CHUNKING = os.getenv("SEMANTIC_CHUNKING", "false").lower() == "true"
    MAX_CHUNK_CHAR = int(os.getenv("MAX_CHUNK_CHAR", "2000"))
    
    # Retrieval
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.3"))
    HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    
    # Cache
    CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
    CACHE_DIR = Path(os.getenv("CACHE_DIR", "./storages/cache"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Embedding instructions
    EMBEDDING_QUERY_INSTRUCTION = os.getenv(
        "EMBEDDING_QUERY_INSTRUCTION",
        "Represent this sentence for searching relevant passages: "
    )
    EMBEDDING_PASSAGE_INSTRUCTION = os.getenv(
        "EMBEDDING_PASSAGE_INSTRUCTION",
        "" 
    )
    
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storages"))
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    UPLOAD_DIR = STORAGE_DIR / "uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    CONTEXT_MAX_RESULTS = int(os.getenv("CONTEXT_MAX_RESULTS", "3"))
    CONTEXT_MAX_TEXT_LENGTH = int(os.getenv("CONTEXT_MAX_TEXT_LENGTH", "800"))
    CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "3"))
    
    # Structured data handling (tables, lists, etc.)
    TABLE_BOOST_MULTIPLIER = float(os.getenv("TABLE_BOOST_MULTIPLIER", "1.5"))
    STRUCTURED_DATA_BOOST_MULTIPLIER = float(os.getenv("STRUCTURED_DATA_BOOST_MULTIPLIER", "1.3"))
    PRESERVE_TABLE_CONTENT = os.getenv("PRESERVE_TABLE_CONTENT", "true").lower() == "true"
    TABLE_CONTEXT_PRIORITY = os.getenv("TABLE_CONTEXT_PRIORITY", "true").lower() == "true"
    
    # HTTP Client & Retry Configuration
    HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
    HTTP_RETRY_DELAY = float(os.getenv("HTTP_RETRY_DELAY", "2.0"))
    HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "10.0"))
    HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "120.0"))
    HTTP_WRITE_TIMEOUT = float(os.getenv("HTTP_WRITE_TIMEOUT", "10.0"))
    HTTP_POOL_TIMEOUT = float(os.getenv("HTTP_POOL_TIMEOUT", "10.0"))
    HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "100"))
    HTTP_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "20"))
    
    # Ingestion timeout (for large files)
    INGEST_TIMEOUT = float(os.getenv("INGEST_TIMEOUT", "600.0"))

    EMBEDDING_PROVIDER_TYPE = os.getenv("EMBEDDING_PROVIDER_TYPE", "http")
    LLM_PROVIDER_TYPE = os.getenv("LLM_PROVIDER_TYPE", "http")
    RERANKER_PROVIDER_TYPE = os.getenv("RERANKER_PROVIDER_TYPE", "http")
config = Config()
