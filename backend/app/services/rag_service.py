import httpx
import asyncio
import os
from typing import Optional, Dict, Any, AsyncIterator
from app.core.config import settings
from app.core.logging import logger
import aiofiles

class RAGService:
    """Client for RAG service HTTP API with connection pooling"""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 600.0,
        stream_timeout: float = 180.0
    ):
        """
        Initialize RAG service client.
        
        Args:
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries in seconds (exponential backoff)
            timeout: Request timeout in seconds
            stream_timeout: Stream request timeout in seconds
        """
        self.base_url = settings.RAG_SERVICE_URL
        self.timeout = httpx.Timeout(timeout, connect=10.0, read=timeout, write=10.0, pool=10.0)
        self.stream_timeout = httpx.Timeout(stream_timeout, connect=10.0, read=stream_timeout, write=10.0, pool=10.0)
        self._client: Optional[httpx.AsyncClient] = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get shared HTTP client with connection pooling"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=10
                )
            )
        return self._client
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic"""
        client = self._get_client()
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "POST":
                    response = await client.post(url, **kwargs)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                return response
                
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}), retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Connection failed after {self.max_retries} attempts: {str(e)}")
                raise
            except httpx.TimeoutException as e:
                logger.error(f"Request timeout: {str(e)}")
                raise
            except httpx.HTTPStatusError as e:
                # Don't retry on HTTP errors (4xx, 5xx)
                raise
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def query(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Query RAG service and get response with sources"""
        try:
            payload = {
                "question": question,
                "filters": filters or {},
                "temperature": temperature,
                "max_tokens": max_tokens,
                "include_sources": include_sources
            }
            
            response = await self._request_with_retry(
                "POST",
                f"{self.base_url}/query",
                json=payload
            )
            
            result = response.json()
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "query_id": result.get("query_id")
            }
        except httpx.HTTPError as e:
            logger.error(f"RAG service HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            raise
    
    async def query_stream(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """Stream response from RAG service"""
        try:
            client = self._get_client()
            payload = {
                "question": question,
                "filters": filters or {},
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            async with client.stream(
                "POST",
                f"{self.base_url}/query",
                json=payload,
                timeout=self.stream_timeout
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
        except httpx.HTTPError as e:
            logger.error(f"RAG stream HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"RAG stream error: {str(e)}")
            raise
    
    async def ingest_document(
        self,
        filepath: str,
        project: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Send document to RAG service for ingestion"""
        try:
            client = self._get_client()
            filename = os.path.basename(filepath)
            
            async with aiofiles.open(filepath, 'rb') as f:
                file_content = await f.read()
            
            files = {'file': (filename, file_content)}
            data = {
                'project': project,
                'language': language
            }
            
            response = await self._request_with_retry(
                "POST",
                f"{self.base_url}/ingest",
                files=files,
                data=data
            )
            
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"RAG ingest HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"RAG ingest error: {str(e)}")
            raise
    
    async def delete_vectors(
        self,
        doc_id: str
    ) -> int:
        """Delete vectors for a document"""
        try:
            response = await self._request_with_retry(
                "DELETE",
                f"{self.base_url}/vectors/{doc_id}"
            )
            
            result = response.json()
            return result.get("deleted_count", 0)
        except httpx.HTTPError as e:
            logger.error(f"RAG delete HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"RAG delete error: {str(e)}")
            raise

rag_service = RAGService()