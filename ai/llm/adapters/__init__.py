"""
LLM Adapters - implementations for different LLM backends.

Each adapter implements BaseLLM interface and provides a unified way to interact
with different LLM providers (Ollama, HuggingFace, etc.)
"""

from ai.llm.adapters.ollama import OllamaAdapter
from ai.llm.adapters.huggingface import HuggingFaceAdapter

__all__ = [
    "OllamaAdapter",
    "HuggingFaceAdapter",
]
