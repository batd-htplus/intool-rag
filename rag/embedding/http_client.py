"""
Legacy HTTP embedding client - DEPRECATED.
Use rag.providers.embedding_provider.HTTPEmbeddingProvider instead.
This wrapper is kept for backward compatibility only.
"""

import warnings
from rag.providers.embedding_provider import HTTPEmbeddingProvider

warnings.warn(
    "HTTPEmbedding is deprecated. Use HTTPEmbeddingProvider from rag.providers instead.",
    DeprecationWarning,
    stacklevel=2
)

class HTTPEmbedding(HTTPEmbeddingProvider):
    """Legacy wrapper - redirects to HTTPEmbeddingProvider"""
    pass

