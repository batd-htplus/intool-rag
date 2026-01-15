"""
LLM Factory - selects and initializes appropriate adapter based on environment.

This is the single point where adapter selection happens.
Core application never knows which adapter is being used.
"""
import os
from ai.logging import logger
from ai.llm.base import BaseLLM


def get_llm() -> BaseLLM:
    """
    Factory function to get LLM instance based on environment config.
    
    Environment variables:
    - LLM_BACKEND: "ollama" or "huggingface" (default: "ollama")
    - LLM_MODEL: model name (required)
    - OLLAMA_BASE_URL: for Ollama adapter
    - LLM_DEVICE: for HuggingFace adapter
    
    Returns:
        BaseLLM instance (either OllamaAdapter or HuggingFaceAdapter)
    
    Raises:
        ValueError: If backend is invalid or model loading fails
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    model = os.getenv("LLM_MODEL", "")
    
    if backend == "ollama":
        try:
            from ai.llm.adapters.ollama import OllamaAdapter
            return OllamaAdapter()
        except ImportError as e:
            raise RuntimeError(f"Failed to import OllamaAdapter: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama adapter: {str(e)}")
    
    elif backend == "huggingface":
        try:
            from ai.llm.adapters.huggingface import HuggingFaceAdapter
            return HuggingFaceAdapter()
        except ImportError as e:
            raise RuntimeError(f"Failed to import HuggingFaceAdapter: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace adapter: {str(e)}")
    
    else:
        raise ValueError(
            f"Unknown LLM backend: {backend}. "
            f"Supported: 'ollama', 'huggingface'"
        )
