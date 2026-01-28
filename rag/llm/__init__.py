# Local LLM Service Integration
# Provides LLM and embedding services directly within RAG service
# No external HTTP calls needed

from rag.llm.llm_service import get_llm, is_llm_loaded
from rag.llm.embedding_service import get_embedding_model, is_embedding_loaded

__all__ = [
    "get_llm",
    "is_llm_loaded",
    "get_embedding_model",
    "is_embedding_loaded",
]

