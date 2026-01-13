from typing import AsyncIterator
import httpx
from model_service.config import config
from model_service.logging import logger
import os

class OllamaLLM:
    """LLM via Ollama API"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        logger.info(f"Using Ollama at {self.base_url} with model {self.model}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response from prompt via Ollama (sync wrapper)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._generate_async(prompt, temperature, max_tokens))
    
    async def _generate_async(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response from prompt via Ollama"""
        try:
            temperature = temperature or config.LLM_TEMPERATURE
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens or config.LLM_MAX_TOKENS
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {str(e)}")
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
        """Stream response generation via Ollama"""
        try:
            temperature = temperature or config.LLM_TEMPERATURE
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens or config.LLM_MAX_TOKENS
                        }
                    }
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.HTTPError as e:
            logger.error(f"Ollama stream HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Stream generation error: {str(e)}")
            raise

