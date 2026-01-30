from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):

    @abstractmethod
    async def embed_single(self, text: str, instruction: str | None = None) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(self, texts: List[str], instruction: str | None = None) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError
