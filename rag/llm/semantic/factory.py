import os
from rag.logging import logger
from .base import SemanticAnalyzer


_ANALYZER: SemanticAnalyzer | None = None


def get_semantic_analyzer() -> SemanticAnalyzer:
    """
    Factory for semantic analyzer.

    Priority:
    1. Gemini (cloud, best quality)
    2. Ollama (local fallback)
    """
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER

    if os.getenv("GEMINI_API_KEY"):
        try:
            from rag.providers.gemini.semantic import GeminiSemanticAnalyzer
            from rag.providers.base import ProviderConfig
            
            config = ProviderConfig(api_key=os.getenv("GEMINI_API_KEY"))
            logger.info("[SEMANTIC] Using GeminiSemanticAnalyzer")
            _ANALYZER = GeminiSemanticAnalyzer(config)
            return _ANALYZER
        except Exception as e:
            logger.warning(f"[SEMANTIC] Gemini unavailable: {e}")

    try:
        from rag.providers.ollama.semantic import OllamaSemanticAnalyzer
        logger.info("[SEMANTIC] Using OllamaSemanticAnalyzer")
        _ANALYZER = OllamaSemanticAnalyzer()
        return _ANALYZER
    except Exception as e:
        raise RuntimeError(f"No semantic analyzer available. Fallback failed: {e}")
