"""
Integration Guide: Page-Aware RAG API
=====================================

This module shows how to integrate the new page-aware RAG system
with existing FastAPI endpoints.

Replace old endpoints with these new page-aware versions.
"""

from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import tempfile
from pathlib import Path

from rag.ingest.page_pipeline import PageAwareIngestPipeline
from rag.query.page_retriever import retrieve_and_rank_pages
from rag.query.page_response import build_rag_prompt, create_page_aware_response
from rag.core.container import get_container
from rag.logging import logger

router = APIRouter()

# ===== Request/Response Models =====

class IngestRequest(BaseModel):
    """Document ingestion request"""
    project: str
    language: Optional[str] = "en"
    doc_id: Optional[str] = None


class IngestResponse(BaseModel):
    """Document ingestion response"""
    success: bool
    doc_id: str
    source_file: str
    project: str
    total_pages: int
    total_chunks: int
    vectors_indexed: int
    processing_time_seconds: float


class QueryRequest(BaseModel):
    """RAG query request"""
    question: str
    project: Optional[str] = None
    top_pages: int = 5
    max_context_length: int = 8000


class SourceReference(BaseModel):
    """Source reference for citation"""
    page: int
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    title: Optional[str] = None
    source_file: Optional[str] = None
    relevance_score: float


class QueryResponse(BaseModel):
    """RAG query response"""
    answer: str
    sources: List[SourceReference]
    confidence: str  # "high", "medium", "low"


# ===== Endpoints =====

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    project: str = Query(...),
    language: Optional[str] = Query("en"),
    doc_id: Optional[str] = Query(None),
):
    """
    Ingest document using 3-phase page-aware pipeline.
    
    - PHASE 1: Load PDF per page, normalize text
    - PHASE 2: Build PageIndex, create chunks
    - PHASE 3: Embed chunks, index vectors
    
    Args:
        file: PDF file to ingest
        project: Project name
        language: Document language
        doc_id: Optional document ID
        
    Returns:
        Ingestion result with doc_id and chunk count
    """
    doc_id = doc_id or str(uuid.uuid4())
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / file.filename
            content = await file.read()
            filepath.write_bytes(content)
            
            pipeline = PageAwareIngestPipeline()
            result = await pipeline.ingest_document(
                filepath=str(filepath),
                language=language,
                doc_id=doc_id,
            )
            
            logger.info(f"✓ Ingested {file.filename}: {result}")
            
            return IngestResponse(
                project=project,
                **result
            )
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using page-aware RAG.
    
    Process:
    1. Embed query
    2. Retrieve top-K chunks
    3. Group chunks by page
    4. Rank pages by relevance
    5. Select top N pages
    6. Assemble context
    7. Generate answer with LLM
    8. Format response with citations
    
    Args:
        request: Query request with question
        
    Returns:
        Answer with source citations
    """
    try:
        logger.info(f"Querying: {request.question}")
        
        ranked_pages = await retrieve_and_rank_pages(
            query=request.question,
            project=request.project,
            top_pages=request.top_pages,
        )
        
        if not ranked_pages:
            return QueryResponse(
                answer="I could not find relevant information to answer your question.",
                sources=[],
                confidence="low"
            )
        
        prompt = build_rag_prompt(
            question=request.question,
            ranked_pages=ranked_pages,
            max_context_length=request.max_context_length,
        )
        
        llm_provider = get_container().get_llm_provider()
        answer = await llm_provider.generate(prompt)
        
        response_dict = create_page_aware_response(answer, ranked_pages)
        
        sources = [
            SourceReference(**src)
            for src in response_dict["sources"]["primary_sources"]
        ]
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=response_dict["confidence"]
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """
    List ingested documents with metadata.
    
    Returns:
        List of documents with page count and chunk count
    """
    try:
        from rag.storage.file_storage import FileStorageManager
        from rag.config import config
        storage = FileStorageManager(config.STORAGE_DIR)
        docs = await storage.get_unique_documents()
        
        return {
            "documents": docs,
            "total": len(docs)
        }
    
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document_info(doc_id: str):
    """
    Get detailed information about a document.
    
    Returns:
        - Document metadata
        - PageIndex (structure)
        - Chunk statistics
    """
    try:
        # Load PageIndex
        page_index_path = Path(f"/path/to/page_indexes/{doc_id}.json")
        
        if not page_index_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        from rag.core.page_index import PageIndex
        page_index = PageIndex.load_from_file(str(page_index_path))
        
        return {
            "doc_id": doc_id,
            "page_count": page_index.get_page_count(),
            "pages": [
                {
                    "page": entry.page,
                    "chapter": entry.chapter,
                    "section": entry.section,
                    "title": entry.title,
                }
                for entry in page_index.get_all_pages()[:10]
            ],
            "source_filename": page_index.source_filename,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "system": "page-aware-rag",
        "version": "2.0",
    }
