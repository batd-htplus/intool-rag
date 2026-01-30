from typing import List
from .factory import get_embedding_provider


async def embed(text: str) -> List[float]:
    return await get_embedding_provider().embed_single(text)


async def embed_batch(texts: List[str]) -> List[List[float]]:
    return await get_embedding_provider().embed_batch(texts)


def embedding_dim() -> int:
    return get_embedding_provider().dimension()
