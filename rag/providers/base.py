from abc import ABC, abstractmethod
from typing import List, Optional

from dataclasses import dataclass

@dataclass
class ProviderConfig:
    api_key: Optional[str] = None
    model: Optional[str] = None
    
class BaseProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config

# ---------- Embeddings ----------
from rag.llm.embeddings.base import EmbeddingProvider

# ---------- Semantic analysis ----------
class SemanticProvider(ABC):
    @abstractmethod
    async def analyze_document(self, prompt: str) -> str:
        """
        MUST return RAW JSON STRING (no markdown)
        """
        pass


# ---------- Text generation ----------
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass
