
import os
from rag.logging import logger
from rag.llm.base import BaseLLM

_LLM_INSTANCE: BaseLLM | None = None

def get_llm() -> BaseLLM:
    """
    Factory to get the configured LLM provider.
    
    Priority:
    1. Gemini (if GEMINI_API_KEY is set)
    2. Ollama (default fallback)
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE
        
    # 1. Try Gemini
    if os.getenv("GEMINI_API_KEY"):
        try:
            from rag.providers.gemini.llm import GeminiLLM
            from rag.providers.base import ProviderConfig
            config = ProviderConfig(
                api_key=os.getenv("GEMINI_API_KEY"),
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            )
            
            _LLM_INSTANCE = GeminiLLM(config)
            logger.info(f"Loaded LLM Provider: Gemini ({config.model})")
            return _LLM_INSTANCE
        except Exception as e:
            logger.warning(f"Failed to load Gemini LLM: {e}")

    # 2. Fallback to Ollama
    try:
        from rag.providers.ollama.llm import OllamaLLMProvider
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        _LLM_INSTANCE = OllamaLLMProvider(model=model)
        logger.info(f"Loaded LLM Provider: Ollama ({model})")
        return _LLM_INSTANCE
    except Exception as e:
        logger.error(f"Failed to load Ollama LLM: {e}")
        raise RuntimeError("No LLM provider available") from e
