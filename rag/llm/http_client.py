"""
Legacy HTTP LLM client - DEPRECATED.
Use rag.providers.llm_provider.HTTPLLMProvider instead.
This wrapper is kept for backward compatibility only.
"""

import warnings
from rag.providers.llm_provider import HTTPLLMProvider

warnings.warn(
    "HTTPLLM is deprecated. Use HTTPLLMProvider from rag.providers instead.",
    DeprecationWarning,
    stacklevel=2
)

class HTTPLLM(HTTPLLMProvider):
    """Legacy wrapper - redirects to HTTPLLMProvider"""
    pass

