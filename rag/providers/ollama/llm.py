import httpx
from rag.config import config
from rag.logging import logger

from typing import AsyncIterator
from rag.llm.base import LLM

class OllamaLLMProvider(LLM):
    """
    Low-level LLM text generation via Ollama HTTP API.
    """

    def __init__(self, model: str):
        self.model = model
        self.base_url = config.LLM_BASE_URL

    async def generate(
        self, 
        prompt: str, 
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if temperature: payload["options"] = {"temperature": temperature}

        async with httpx.AsyncClient(timeout=120) as client:
            logger.info(f"Ollama request to {self.base_url}/api/generate")
            
            try:
                r = await client.post(f"{self.base_url}/api/generate", json=payload)
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except Exception as e:
                logger.error(f"Ollama request failed: {e}")
                raise

    async def generate_stream(
        self, 
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        if temperature: payload["options"] = {"temperature": temperature}
        
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                async for line in response.aiter_lines():
                    if not line: continue
                    import json
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                    except:
                        pass

    def is_ready(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "provider": "ollama",
            "model": self.model
        }
