"""
LLM service wrapper - provides singleton LLM instance.

Core application gets LLM through this service.
Actual adapter selection is handled by factory (ai/llm/factory.py).
"""
import threading
from ai.logging import logger
from ai.llm.factory import get_llm as create_llm

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

