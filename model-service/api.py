"""
Model Service API
Serves LLM and Embedding models via HTTP API
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from contextlib import asynccontextmanager
from model_service.logging import logger
from model_service.embedding_service import get_embedding_model
from model_service.llm_service import get_llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    logger.info("Loading models on startup...")
    try:
        logger.info("Loading embedding model...")
        embedding_model = get_embedding_model()
        logger.info("✓ Embedding model loaded")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
    
    try:
        logger.info("Loading LLM model...")
        llm = get_llm()
        logger.info("✓ LLM model loaded")
    except Exception as e:
        logger.error(f"Failed to load LLM model: {e}")
    
    logger.info("✓ All models loaded and ready")
    
    yield  # App runs here
    
    # Shutdown
    logger.info("Shutting down model service...")

app = FastAPI(title="Model Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: Optional[str] = None

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class GenerateResponse(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str
    embedding_loaded: bool
    llm_loaded: bool

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        embedding_model = get_embedding_model()
        llm = get_llm()
        return {
            "status": "healthy",
            "embedding_loaded": embedding_model is not None,
            "llm_loaded": llm is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "embedding_loaded": False,
            "llm_loaded": False
        }

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Embed texts into vectors"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts list cannot be empty")
        
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise HTTPException(status_code=503, detail="Embedding model not loaded")
        
        embeddings = embedding_model.embed(request.texts, instruction=request.instruction)
        
        return {"embeddings": embeddings}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class EmbedSingleRequest(BaseModel):
    text: str
    instruction: Optional[str] = None

@app.post("/embed/single", response_model=Dict[str, Any])
async def embed_single(request: EmbedSingleRequest):
    """Embed single text into vector"""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="text cannot be empty")
        
        embedding_model = get_embedding_model()
        if embedding_model is None:
            raise HTTPException(status_code=503, detail="Embedding model not loaded")
        
        embedding = embedding_model.embed_single(request.text, instruction=request.instruction)
        
        return {"embedding": embedding}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt using LLM"""
    try:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="prompt cannot be empty")
        
        llm = get_llm()
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not loaded")
        
        if hasattr(llm, '_generate_async'):
            text = await llm._generate_async(
                request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        else:
            import asyncio
            text = await asyncio.to_thread(
                llm.generate,
                request.prompt,
                request.temperature,
                request.max_tokens
            )
        
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get model service configuration"""
    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        "embedding_device": os.getenv("EMBEDDING_DEVICE", "cpu"),
        "llm_model": os.getenv("LLM_MODEL", "Phi3:mini"),
        "llm_device": os.getenv("LLM_DEVICE", "cpu"),
        "use_ollama": os.getenv("USE_OLLAMA", "false").lower() == "true"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)

