"""
HTTP client for embedding model service
"""
import httpx
from typing import List
from rag.config import config
from rag.logging import logger
import os

class HTTPEmbedding:
    """Embedding via HTTP API (model-service)"""
    
    def __init__(self):
        self.base_url = os.getenv("MODEL_SERVICE_URL", "http://model-service:8002")
        logger.info(f"Using embedding service at {self.base_url}")
    
    def embed(self, texts: List[str], instruction: str = None) -> List[List[float]]:
        """
        Embed list of texts via HTTP API
        
        Args:
            texts: List of texts to embed
            instruction: Optional instruction text (e.g., for query or passage)
        """
        try:
            with httpx.Client(timeout=300.0) as client:
                payload = {"texts": texts}
                if instruction:
                    payload["instruction"] = instruction
                
                response = client.post(
                    f"{self.base_url}/embed",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result["embeddings"]
        except httpx.HTTPError as e:
            logger.error(f"Embedding HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def embed_single(self, text: str, instruction: str = None) -> List[float]:
        """
        Embed single text via HTTP API
        
        Args:
            text: Text to embed
            instruction: Optional instruction text
        """
        try:
            embeddings = self.embed([text], instruction=instruction)
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise

