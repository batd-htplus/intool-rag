"""HTTP-based LLM provider for calling remote model service."""

from typing import Optional, AsyncIterator, Dict, Any
import httpx
import json
import asyncio
from rag.config import config
from rag.logging import logger
from rag.core.exceptions import LLMError, ProviderConnectionError, ProviderTimeoutError
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
        http_client: Optional[httpx.AsyncClient] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ):
        """
        Initialize HTTP LLM provider.
        
        Args:
            base_url: Base URL for LLM service
            http_client: Optional shared HTTP client (for DI)
            max_retries: Maximum retry attempts (defaults to config)
            retry_delay: Base delay between retries in seconds (defaults to config)
        """
        self.base_url = base_url or config.MODEL_SERVICE_URL
        self.client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.HTTP_READ_TIMEOUT,
                connect=config.HTTP_CONNECT_TIMEOUT,
                read=config.HTTP_READ_TIMEOUT,
                write=config.HTTP_WRITE_TIMEOUT,
                pool=config.HTTP_POOL_TIMEOUT
            )
        )
        self._owns_client = http_client is None
        self.max_retries = max_retries or config.HTTP_MAX_RETRIES
        self.retry_delay = retry_delay or config.HTTP_RETRY_DELAY

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt with retry logic.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            ValueError: If prompt is empty
            ProviderTimeoutError: If request times out
            LLMError: For other errors
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        else:
            payload["max_tokens"] = config.LLM_MAX_TOKENS

        payload.update(kwargs)

        return await self._generate_with_retry(payload, retry_count=0)
    
    async def _generate_with_retry(
        self,
        payload: Dict[str, Any],
        retry_count: int = 0
    ) -> str:
        """Generate with retry logic for connection errors."""
        try:
            response = await self.client.post(
                f"{self.base_url}/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            if "text" in result:
                text = result["text"]
                if text:
                    return text
            elif "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0].get("text", "")
                if text:
                    return text

            logger.warning(f"Empty LLM response: {result}")
            raise LLMError("LLM returned empty response")

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"LLM connection error (attempt {retry_count + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                return await self._generate_with_retry(payload, retry_count + 1)
            logger.error(f"LLM connection failed after {self.max_retries + 1} attempts: {str(e)}")
            raise ProviderConnectionError(f"LLM service unavailable: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"LLM generation timeout: {str(e)}")
            raise ProviderTimeoutError(
                "LLM generation timed out. The model may be taking too long to respond. "
                "Please try again or reduce max_tokens."
            )
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if 500 <= status_code < 600 and retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"LLM server error {status_code} (attempt {retry_count + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                return await self._generate_with_retry(payload, retry_count + 1)
            logger.error(f"LLM HTTP error {status_code}: {str(e)}")
            raise LLMError(f"LLM service error: {status_code}")
        except LLMError:
            raise
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            raise LLMError(f"Unexpected error: {str(e)}")

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
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    chunk = data["choices"][0].get("text", "")
                                    if chunk:
                                        yield chunk
                            except Exception as e:
                                logger.warning(f"Error parsing stream chunk: {str(e)}")
                                continue

        except httpx.TimeoutException as e:
            logger.error(f"LLM stream timeout: {str(e)}")
            raise ProviderTimeoutError("LLM stream generation timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM stream HTTP error {e.response.status_code}: {str(e)}")
            raise LLMError(f"LLM stream service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"LLM stream generation error: {str(e)}")
            raise LLMError(f"Unexpected stream error: {str(e)}")

    async def close(self):
        """Close HTTP client (only if we own it, otherwise managed by DI container)."""
        if self._owns_client:
            await self.client.aclose()
