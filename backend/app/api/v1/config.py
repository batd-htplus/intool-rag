from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel
from app.core.config import settings
from app.core.logging import logger
import httpx

router = APIRouter(tags=["config"])

class ModelConfig(BaseModel):
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_device: Optional[str] = None
    llm_device: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    retrieval_top_k: Optional[int] = None

@router.get("/model")
async def get_model_config():
    """Get current model configuration"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.RAG_SERVICE_URL}/config")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=503, detail="RAG service unavailable")
    except Exception as e:
        logger.error(f"Error getting model config: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to RAG service: {str(e)}")

@router.put("/model")
async def update_model_config(config: ModelConfig):
    """Update model configuration"""
    try:
        async with httpx.AsyncClient() as client:
            current_config = await client.get(f"{settings.RAG_SERVICE_URL}/config")
            current = current_config.json() if current_config.status_code == 200 else {}
            
            reload_models = False
            if config.embedding_model and config.embedding_model != current.get("embedding_model"):
                reload_models = True
            if config.llm_model and config.llm_model != current.get("llm_model"):
                reload_models = True
            if config.embedding_device and config.embedding_device != current.get("embedding_device"):
                reload_models = True
            if config.llm_device and config.llm_device != current.get("llm_device"):
                reload_models = True
            
            response = await client.put(
                f"{settings.RAG_SERVICE_URL}/config",
                params={"reload_models": reload_models},
                json=config.dict(exclude_none=True)
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.text
                )
    except httpx.RequestError as e:
        logger.error(f"Error updating model config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to RAG service: {str(e)}"
        )

@router.post("/model/reload")
async def reload_models():
    """Force reload models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{settings.RAG_SERVICE_URL}/config/reload")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.text
                )
    except httpx.RequestError as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to RAG service: {str(e)}"
        )

