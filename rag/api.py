from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag.index import engine
from rag.ingest.pipeline import pipeline
from rag.config import config
from rag.logging import logger
from rag.background_tasks import startup_background_tasks, shutdown_background_tasks
from rag.cache import get_embedding_cache, get_query_cache
from rag.core.container import get_container, shutdown_container
from rag.core.exceptions import RAGError, EmbeddingError, LLMError, RetrievalError
import time
import uuid
import os
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any

app = FastAPI(
    title="RAG Service",
    version="0.2.0",
    description="Optimized RAG Query & Ingest Service with Hybrid Search and Semantic Chunking"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize all services and DI container"""
    try:
        container = get_container()
        container.get_embedding_provider()
        container.get_llm_provider()
        if config.RERANKER_ENABLED:
            container.get_reranker_provider()
    except Exception as e:
        logger.error(f"Failed to initialize providers: {str(e)}")
        raise
    
    await startup_background_tasks()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup all resources"""
    try:
        await shutdown_background_tasks()
        await shutdown_container()
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    include_sources: bool = True

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "service": "RAG"}

@app.get("/ready")
async def readiness():
    """Readiness check"""
    return {
        "status": "ready",
        "components": {
            "embedding": "loaded",
            "llm": "loaded",
            "vector_store": "connected",
        },
    }

@app.post("/query")
async def query(request: QueryRequest):
    """
    Query the RAG engine with optional source tracking.
    
    Uses DI container for:
    - Shared HTTP client (connection pooling)
    - Query result caching
    - Hybrid search (vector + BM25)
    - Optional reranking
    
    Error Handling:
    - EmbeddingError: Failed to embed question
    - RetrievalError: Failed to retrieve documents
    - LLMError: Failed to generate answer
    """
    try:
        if request.stream:
            async def generate():
                try:
                    async for chunk in engine.query_stream(
                        question=request.question,
                        filters=request.filters,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ):
                        yield chunk
                except EmbeddingError as e:
                    logger.error(f"Embedding error: {str(e)}")
                    yield f"ERROR: Failed to embed question: {str(e)}"
                except RetrievalError as e:
                    logger.error(f"Retrieval error: {str(e)}")
                    yield f"ERROR: Failed to retrieve documents: {str(e)}"
                except LLMError as e:
                    logger.error(f"LLM error: {str(e)}")
                    yield f"ERROR: Failed to generate answer: {str(e)}"
            
            return StreamingResponse(generate(), media_type="text/event-stream")

        result = await engine.query(
            question=request.question,
            filters=request.filters,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_sources=request.include_sources,
        )
        
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "query_id": str(uuid.uuid4()),
            "timestamp": int(time.time()),
        }
    except EmbeddingError as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {str(e)}")
    except RetrievalError as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Retrieval error: {str(e)}")
    except LLMError as e:
        logger.error(f"LLM error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"LLM service error: {str(e)}")
    except RAGError as e:
        logger.error(f"RAG error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected query error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    project: str = Form(...),
    language: str = Form("en"),
    async_mode: bool = Form(False, description="Process in background"),
):
    """
    Upload and ingest a document.
    
    Args:
        file: Document file
        project: Project identifier
        language: Document language
        async_mode: If True, process in background; if False, process synchronously
    """
    try:
        upload_dir = Path("/storage/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        
        result = await pipeline.ingest_document(
            filepath=str(file_path),
            project=project,
            language=language,
            async_mode=async_mode
        )
        
        if not async_mode:
            try:
                if file_path.exists():
                    os.remove(file_path)
            except OSError as e:
                logger.warning(f"Could not remove temporary file {file_path}: {e}")
        
        return result
    except EmbeddingError as e:
        logger.error(f"Embedding error during ingestion: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {str(e)}")
    except RAGError as e:
        logger.error(f"RAG error during ingestion: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected ingest error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/vectors/{doc_id}")
async def delete_vectors(doc_id: str):
    """Delete vectors for a document"""
    try:
        from rag.vector_store.qdrant import delete

        deleted_count = await delete(doc_id)
        return {
            "doc_id": doc_id,
            "deleted_count": deleted_count,
        }
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current configuration including optimization settings"""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "embedding_device": config.EMBEDDING_DEVICE,
        "llm_device": config.LLM_DEVICE,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "semantic_chunking": config.SEMANTIC_CHUNKING,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "hybrid_search_enabled": config.HYBRID_SEARCH_ENABLED,
        "reranker_enabled": config.RERANKER_ENABLED,
        "cache_embeddings": config.CACHE_EMBEDDINGS,
        "bm25_weight": config.BM25_WEIGHT,
        "vector_weight": config.VECTOR_WEIGHT,
        "qdrant_hnsw_m": config.QDRANT_HNSW_M,
        "qdrant_hnsw_ef_construct": config.QDRANT_HNSW_EF_CONSTRUCT,
        "qdrant_hnsw_ef_search": config.QDRANT_HNSW_EF_SEARCH,
    }

@app.put("/config")
async def update_config(
    new_config: Dict[str, Any],
    reload_models: bool = Query(False, description="Force reload models after config change")
):
    """Update configuration (runtime only, not persistent)"""
    try:
        model_changed = False
        
        if "embedding_model" in new_config:
            if config.EMBEDDING_MODEL != new_config["embedding_model"]:
                model_changed = True
            config.EMBEDDING_MODEL = new_config["embedding_model"]
        if "llm_model" in new_config:
            if config.LLM_MODEL != new_config["llm_model"]:
                model_changed = True
            config.LLM_MODEL = new_config["llm_model"]
        if "reranker_model" in new_config:
            config.RERANKER_MODEL = new_config["reranker_model"]
        
        # Device settings
        if "embedding_device" in new_config:
            if config.EMBEDDING_DEVICE != new_config["embedding_device"]:
                model_changed = True
            config.EMBEDDING_DEVICE = new_config["embedding_device"]
        if "llm_device" in new_config:
            if config.LLM_DEVICE != new_config["llm_device"]:
                model_changed = True
            config.LLM_DEVICE = new_config["llm_device"]
        
        # Generation parameters
        if "temperature" in new_config:
            config.LLM_TEMPERATURE = float(new_config["temperature"])
        if "max_tokens" in new_config:
            config.LLM_MAX_TOKENS = int(new_config["max_tokens"])
        
        # Chunking strategy
        if "chunk_size" in new_config:
            config.CHUNK_SIZE = int(new_config["chunk_size"])
        if "chunk_overlap" in new_config:
            config.CHUNK_OVERLAP = int(new_config["chunk_overlap"])
        if "semantic_chunking" in new_config:
            config.SEMANTIC_CHUNKING = new_config["semantic_chunking"]
        
        # Retrieval optimization
        if "retrieval_top_k" in new_config:
            config.RETRIEVAL_TOP_K = int(new_config["retrieval_top_k"])
        if "hybrid_search_enabled" in new_config:
            config.HYBRID_SEARCH_ENABLED = new_config["hybrid_search_enabled"]
        if "reranker_enabled" in new_config:
            config.RERANKER_ENABLED = new_config["reranker_enabled"]
        if "bm25_weight" in new_config:
            config.BM25_WEIGHT = float(new_config["bm25_weight"])
        if "vector_weight" in new_config:
            config.VECTOR_WEIGHT = float(new_config["vector_weight"])
        
        # Cache settings
        if "cache_embeddings" in new_config:
            config.CACHE_EMBEDDINGS = new_config["cache_embeddings"]
        
        # Qdrant HNSW tuning
        if "qdrant_hnsw_m" in new_config:
            config.QDRANT_HNSW_M = int(new_config["qdrant_hnsw_m"])
        if "qdrant_hnsw_ef_construct" in new_config:
            config.QDRANT_HNSW_EF_CONSTRUCT = int(new_config["qdrant_hnsw_ef_construct"])
        if "qdrant_hnsw_ef_search" in new_config:
            config.QDRANT_HNSW_EF_SEARCH = int(new_config["qdrant_hnsw_ef_search"])
        
        
        if model_changed or reload_models:
            try:
                import rag.index as index_module
                index_module._llm = None
            except Exception as e:
                logger.warning(f"Could not clear LLM client: {str(e)}")
        
        return {
            "status": "updated",
            "models_reloaded": model_changed or reload_models,
            "config": await get_config()
        }
    except Exception as e:
        logger.error(f"Config update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache(cache_type: str = Query("all", description="Cache type: embeddings, queries, or all")):
    """Clear cache (embeddings, queries, or both)"""
    try:
        if cache_type in ["embeddings", "all"]:
            emb_cache = get_embedding_cache()
            emb_cache.clear()
        
        if cache_type in ["queries", "all"]:
            query_cache = get_query_cache()
            query_cache.clear()
        
        return {
            "status": "success",
            "cleared": cache_type
        }
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
