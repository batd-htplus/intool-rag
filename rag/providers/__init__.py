"""
Provider abstraction layer for pluggable models.
Supports easy switching between different embedding and LLM models.
"""

from .base import (
    EmbeddingProvider,
    LLMProvider,
)
from .llm_provider import LocalLLMProvider
from .gemini_embedding import GeminiEmbeddingProvider
from .langchain_wrapper import HuggingFaceEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LocalLLMProvider",
    "GeminiEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
]
