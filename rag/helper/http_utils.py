"""Shared HTTP utilities for providers (connection pooling, retry logic)."""

import httpx
import asyncio
from typing import Dict, Any, Optional, Callable, TypeVar
from rag.config import config
from rag.logging import logger
from rag.core.exceptions import (
    ProviderConnectionError,
    ProviderTimeoutError,
)

T = TypeVar("T")

_shared_http_client: Optional[httpx.AsyncClient] = None


def get_shared_http_client() -> httpx.AsyncClient:
    """Get or create global shared HTTP client with connection pooling."""
    global _shared_http_client
    
    if _shared_http_client is None:
        _shared_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.HTTP_READ_TIMEOUT,
                connect=config.HTTP_CONNECT_TIMEOUT,
                read=config.HTTP_READ_TIMEOUT,
                write=config.HTTP_WRITE_TIMEOUT,
                pool=config.HTTP_POOL_TIMEOUT
            )
        )
    
    return _shared_http_client


async def close_shared_http_client():
    """Close global shared HTTP client."""
    global _shared_http_client
    
    if _shared_http_client is not None:
        await _shared_http_client.aclose()
        _shared_http_client = None


async def http_request_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    service_name: str = "Service",
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    retry_count: int = 0,
) -> Dict[str, Any]:
    """
    Make HTTP request with exponential backoff retry for transient errors.
    
    Args:
        client: HTTP client to use
        url: Request URL
        payload: Request payload
        service_name: Service name for logging
        max_retries: Max retry attempts (defaults to config)
        retry_delay: Base delay between retries (defaults to config)
        retry_count: Internal - current retry attempt
        
    Returns:
        Response JSON data
        
    Raises:
        ProviderConnectionError: Connection failed after retries
        ProviderTimeoutError: Request timed out
    """
    max_retries = max_retries or config.HTTP_MAX_RETRIES
    retry_delay = retry_delay or config.HTTP_RETRY_DELAY
    
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        if retry_count < max_retries:
            delay = retry_delay * (2 ** retry_count)
            logger.warning(
                f"{service_name} connection error "
                f"(attempt {retry_count + 1}/{max_retries + 1}), "
                f"retrying in {delay}s: {str(e)}"
            )
            await asyncio.sleep(delay)
            return await http_request_with_retry(
                client, url, payload, service_name, max_retries, retry_delay, retry_count + 1
            )
        logger.error(f"{service_name} connection failed after {max_retries + 1} attempts: {str(e)}")
        raise ProviderConnectionError(f"{service_name} unavailable: {str(e)}")
    except httpx.TimeoutException as e:
        if retry_count < max_retries:
            delay = retry_delay * (2 ** retry_count)
            logger.warning(
                f"{service_name} timeout "
                f"(attempt {retry_count + 1}/{max_retries + 1}), "
                f"retrying in {delay}s"
            )
            await asyncio.sleep(delay)
            return await http_request_with_retry(
                client, url, payload, service_name, max_retries, retry_delay, retry_count + 1
            )
        logger.error(f"{service_name} timeout after {max_retries + 1} attempts")
        raise ProviderTimeoutError(f"{service_name} timed out. Please try again.")
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if 500 <= status_code < 600 and retry_count < max_retries:
            delay = retry_delay * (2 ** retry_count)
            logger.warning(
                f"{service_name} server error {status_code} "
                f"(attempt {retry_count + 1}/{max_retries + 1}), "
                f"retrying in {delay}s"
            )
            await asyncio.sleep(delay)
            return await http_request_with_retry(
                client, url, payload, service_name, max_retries, retry_delay, retry_count + 1
            )
        logger.error(f"{service_name} HTTP error {status_code}")
        raise
    except Exception as e:
        logger.error(f"{service_name} error: {str(e)}")
        raise
