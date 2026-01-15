import os
from pathlib import Path

class Config:
    """RAG service configuration"""
    
    MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://ai-service:8002")
    
    EMBEDDING_PROVIDER_TYPE = os.getenv("EMBEDDING_PROVIDER_TYPE", "http")
    LLM_PROVIDER_TYPE = os.getenv("LLM_PROVIDER_TYPE", "http")
    RERANKER_PROVIDER_TYPE = os.getenv("RERANKER_PROVIDER_TYPE", "http")
    
    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    
    # LLM
    LLM_MODEL = os.getenv("LLM_MODEL", "Phi3:mini")
    LLM_DEVICE = os.getenv("LLM_DEVICE", "cpu")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
    LLM_RELEVANCE_THRESHOLD = float(os.getenv("LLM_RELEVANCE_THRESHOLD", "0.4"))
    
    # Reranker
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "10"))
    
    # Vector Store
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))
    
    # Qdrant HNSW optimization
    QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "16"))
    QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "200"))
    QDRANT_HNSW_EF_SEARCH = int(os.getenv("QDRANT_HNSW_EF_SEARCH", "100"))
    
    # Chunking strategy
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
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
    CACHE_DIR = Path(os.getenv("CACHE_DIR", "/storage/cache"))
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
    
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/storage"))
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    CONTEXT_MAX_RESULTS = int(os.getenv("CONTEXT_MAX_RESULTS", "3"))
    CONTEXT_MAX_TEXT_LENGTH = int(os.getenv("CONTEXT_MAX_TEXT_LENGTH", "800"))
    CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "3"))

config = Config()
