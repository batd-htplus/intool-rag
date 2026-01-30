
"""
Provider abstraction layer for pluggable models.
Supports easy switching between different embedding and LLM models.
"""

from .base import (
    EmbeddingProvider,
    LLMProvider,
)

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
] 
