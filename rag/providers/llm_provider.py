"""HTTP-based LLM provider for calling remote model service."""

from typing import Optional, AsyncIterator
import httpx
from rag.config import config
from rag.logging import logger
from .base import LLMProvider


class HTTPLLMProvider(LLMProvider):
    """LLM provider that uses HTTP calls to model service.
    
    Can use either:
    1. Shared HTTP client (recommended for DI) - via DI container
    2. Own HTTP client (standalone mode) - creates new client
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        self.base_url = base_url or config.MODEL_SERVICE_URL
        self.client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0, read=120.0, write=10.0, pool=10.0)
        )
        self._owns_client = http_client is None

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        try:
            payload = {
                "prompt": prompt,
                "temperature": temperature,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            else:
                payload["max_tokens"] = config.LLM_MAX_TOKENS

            payload.update(kwargs)

            response = await self.client.post(
                f"{self.base_url}/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            if "text" in result:
                return result["text"]
            elif "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("text", "")

            return ""

        except httpx.TimeoutException as e:
            logger.error(f"LLM generation timeout: {str(e)}")
            raise RuntimeError("LLM generation timed out. The model may be taking too long to respond. Please try again or reduce max_tokens.")
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error {e.response.status_code}: {str(e)}")
            raise RuntimeError(f"LLM service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate text from prompt with streaming."""
        try:
            payload = {
                "prompt": prompt,
                "temperature": temperature,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            else:
                payload["max_tokens"] = config.LLM_MAX_TOKENS

            payload.update(kwargs)

            async with self.client.stream(
                "POST",
                f"{self.base_url}/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str:
                            try:
                                import json

                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    chunk = data["choices"][0].get("text", "")
                                    if chunk:
                                        yield chunk
                            except Exception as e:
                                logger.warning(f"Error parsing stream chunk: {str(e)}")
                                continue

        except Exception as e:
            logger.error(f"LLM stream generation error: {str(e)}")
            raise

    async def close(self):
        """Close HTTP client (only if we own it, otherwise managed by DI container)."""
        if self._owns_client:
            await self.client.aclose()
