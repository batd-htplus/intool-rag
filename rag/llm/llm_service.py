"""
Local LLM Service - provides singleton LLM instance.

Integrates Ollama or HuggingFace LLM directly within RAG service.
No external HTTP dependencies.
"""
import threading
from rag.logging import logger
from rag.llm.factory import get_llm as create_llm

_llm = None
_llm_lock = threading.Lock()

def get_llm():
    """
    Get LLM instance (lazy loaded, singleton, thread-safe).
    
    Returns:
        BaseLLM instance (adapter chosen by factory)
        
    Raises:
        RuntimeError: If LLM initialization fails
    """
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                try:
                    _llm = create_llm()
                except Exception as e:
                    logger.error(f"Failed to initialize LLM: {e}")
                    raise
    return _llm

def is_llm_loaded():
    """Check if LLM is loaded (non-blocking)"""
    return _llm is not None


