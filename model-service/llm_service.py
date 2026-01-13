"""
LLM wrapper for model service
"""
import os
from model_service.logging import logger

_llm = None

def get_llm():
    """Get LLM instance (lazy loaded, singleton)"""
    global _llm
    if _llm is None:
        try:
            if os.getenv("USE_OLLAMA", "false").lower() == "true":
                from model_service.llm.ollama import OllamaLLM
                logger.info("Loading Ollama LLM...")
                _llm = OllamaLLM()
            else:
                from model_service.llm.qwen import QwenLLM
                logger.info("Loading Qwen LLM...")
                _llm = QwenLLM()
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}", exc_info=True)
            raise
    return _llm

