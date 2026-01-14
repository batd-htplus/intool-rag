"""
LLM service wrapper - provides singleton LLM instance.

Core application gets LLM through this service.
Actual adapter selection is handled by factory (model-service/llm/factory.py).
"""
from model_service.logging import logger
from model_service.llm.factory import get_llm as create_llm

_llm = None

def get_llm():
    """
    Get LLM instance (lazy loaded, singleton).
    
    Returns:
        BaseLLM instance (adapter chosen by factory)
        
    Raises:
        RuntimeError: If LLM initialization fails
    """
    global _llm
    if _llm is None:
        try:
            _llm = create_llm()
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise
    return _llm

