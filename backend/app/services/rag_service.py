import httpx
from typing import Optional, Dict, Any, AsyncIterator, List
from app.core.config import settings
from app.core.logging import logger

class RAGService:
    """Client for RAG service HTTP API"""
    
    def __init__(self):
        self.base_url = settings.RAG_SERVICE_URL
        self.timeout = 120
        self.stream_timeout = 180
    
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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "question": question,
                    "filters": filters or {},
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "include_sources": include_sources
                }
                
                response = await client.post(
                    f"{self.base_url}/query",
                    json=payload
                )
                response.raise_for_status()
                
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
            async with httpx.AsyncClient(timeout=self.stream_timeout) as client:
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
                    json=payload
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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(filepath, 'rb') as f:
                    files = {'file': (filepath.split('/')[-1], f)}
                    data = {
                        'project': project,
                        'language': language
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/ingest",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    
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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.base_url}/vectors/{doc_id}"
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("deleted_count", 0)
        except httpx.HTTPError as e:
            logger.error(f"RAG delete HTTP error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"RAG delete error: {str(e)}")
            raise

rag_service = RAGService()