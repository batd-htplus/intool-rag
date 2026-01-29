# rag/ingest/gemini/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiSemanticConfig:
    """
    Configuration for Gemini-based semantic summarization.
    """

    model: str = "gemini-3-flash-preview"

    temperature: float = 0.0
    max_output_tokens: int = 16384

    timeout_seconds: int = 120
