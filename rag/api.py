from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rag.index import engine
from rag.ingest.pipeline import pipeline
from rag.config import config
from rag.logging import logger
import time
import uuid
import os
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any

app = FastAPI(
    title="RAG Service",
    version="0.1.0",
    description="RAG Query & Ingest Service"
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
    logger.info("RAG Service starting...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG Service stopping...")

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
async def query(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    include_sources: bool = True,
):
    """
    Query the RAG engine with optional source tracking.
    """
    try:
        if stream:
            async def generate():
                async for chunk in engine.query_stream(
                    question=question,
                    filters=filters,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk
            return StreamingResponse(generate(), media_type="text/event-stream")

        result = await engine.query(
            question=question,
            filters=filters,
            temperature=temperature,
            max_tokens=max_tokens,
            include_sources=include_sources,
        )
        
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "query_id": str(uuid.uuid4()),
            "timestamp": int(time.time()),
        }
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    project: str = Form(...),
    language: str = Form("en"),
):
    """Upload and ingest a document"""
    try:
        upload_dir = Path("/storage/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        result = await pipeline.ingest_document(
            filepath=str(file_path),
            project=project,
            language=language,
        )
        
        try:
            if file_path.exists():
                os.remove(file_path)
        except OSError as e:
            logger.warning(f"Could not remove temporary file {file_path}: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Get current configuration"""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "embedding_device": config.EMBEDDING_DEVICE,
        "llm_device": config.LLM_DEVICE,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
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
        if "embedding_device" in new_config:
            if config.EMBEDDING_DEVICE != new_config["embedding_device"]:
                model_changed = True
            config.EMBEDDING_DEVICE = new_config["embedding_device"]
        if "llm_device" in new_config:
            if config.LLM_DEVICE != new_config["llm_device"]:
                model_changed = True
            config.LLM_DEVICE = new_config["llm_device"]
        if "temperature" in new_config:
            config.LLM_TEMPERATURE = float(new_config["temperature"])
        if "max_tokens" in new_config:
            config.LLM_MAX_TOKENS = int(new_config["max_tokens"])
        if "chunk_size" in new_config:
            config.CHUNK_SIZE = int(new_config["chunk_size"])
        if "chunk_overlap" in new_config:
            config.CHUNK_OVERLAP = int(new_config["chunk_overlap"])
        if "retrieval_top_k" in new_config:
            config.RETRIEVAL_TOP_K = int(new_config["retrieval_top_k"])
        
        logger.info(f"Configuration updated: {new_config}")
        
        if (model_changed or reload_models):
            logger.info("Clearing HTTP clients to reconnect to model-service...")
            try:
                import rag.query.retriever as retriever_module
                import rag.index as index_module
                
                retriever_module._embedding_model = None
                index_module._llm = None
                
                logger.info("HTTP clients cleared. They will reconnect on next use.")
            except Exception as e:
                logger.warning(f"Could not clear clients: {str(e)}")
        
        return {
            "status": "updated",
            "models_reloaded": model_changed or reload_models,
            "config": {
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_model": config.LLM_MODEL,
                "embedding_device": config.EMBEDDING_DEVICE,
                "llm_device": config.LLM_DEVICE,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "retrieval_top_k": config.RETRIEVAL_TOP_K,
            }
        }
    except Exception as e:
        logger.error(f"Config update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/reload")
async def reload_models():
    """Clear HTTP clients to reconnect to model-service"""
    try:
        import rag.query.retriever as retriever_module
        import rag.index as index_module
        
        retriever_module._embedding_model = None
        index_module._llm = None
        
        logger.info("HTTP clients cleared. They will reconnect to model-service on next use.")
        
        return {
            "status": "success",
            "message": "HTTP clients cleared. They will reconnect to model-service on next use."
        }
    except Exception as e:
        logger.error(f"Reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rag.api:app",
        host="0.0.0.0",
        port=8001,
        reload=config.LOG_LEVEL == "DEBUG",
    )
