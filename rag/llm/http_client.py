"""
HTTP client for LLM model service
"""
import httpx
from typing import AsyncIterator
from rag.config import config
from rag.logging import logger
import os

class HTTPLLM:
    """LLM via HTTP API (model-service)"""
    
    def __init__(self):
        self.base_url = os.getenv("MODEL_SERVICE_URL", "http://model-service:8002")
        logger.info(f"Using LLM service at {self.base_url}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response from prompt via HTTP API"""
        try:
            temperature = temperature or config.LLM_TEMPERATURE
            max_tokens = max_tokens or config.LLM_MAX_TOKENS
            
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["text"]
        except httpx.HTTPError as e:
            logger.error(f"LLM HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> AsyncIterator[str]:
        """Stream response generation via HTTP API"""
        response = self.generate(prompt, temperature, max_tokens)
        
        for word in response.split():
            yield word + " "

