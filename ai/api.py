"""
Model Service API
Serves LLM and Embedding models via HTTP API with concurrency control
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import asyncio
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.logging import logger
from ai.config import config
from ai.embedding_service import get_embedding_model, is_embedding_loaded
from ai.llm_service import get_llm, is_llm_loaded

embed_semaphore = asyncio.Semaphore(config.EMBED_CONCURRENCY)
llm_semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)

executor = ThreadPoolExecutor(max_workers=config.THREADPOOL_MAX_WORKERS)

@asynccontextmanager
async def semaphore_context(semaphore: asyncio.Semaphore, timeout: float):
    """Context manager to acquire and release semaphore with timeout"""
    acquired = False
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        acquired = True
        yield
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Service busy, please try again later"
        )
    finally:
        if acquired:
            semaphore.release()

_embedding_model_cache = None
_llm_model_cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global _embedding_model_cache, _llm_model_cache
    try:
        _embedding_model_cache = get_embedding_model()
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
    
    try:
        _llm_model_cache = get_llm()
    except Exception as e:
        logger.error(f"Failed to load LLM model: {e}")
    
    yield
    
    executor.shutdown(wait=False)

app = FastAPI(title="Model Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: Optional[str] = None

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class EmbedSingleRequest(BaseModel):
    text: str
    instruction: Optional[str] = None

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
    """Health check endpoint (non-blocking, doesn't initialize models)"""
    return {
        "status": "healthy" if (is_embedding_loaded() and is_llm_loaded()) else "unhealthy",
        "embedding_loaded": is_embedding_loaded(),
        "llm_loaded": is_llm_loaded()
    }

def _embed_batch(texts: List[str], instruction: Optional[str] = None) -> List[List[float]]:
    """Blocking embedding function to run in threadpool"""
    model = _embedding_model_cache or get_embedding_model()
    if model is None:
        raise RuntimeError("Embedding model not loaded")
    return model.embed(texts, instruction=instruction)

def _embed_single(text: str, instruction: Optional[str] = None) -> List[float]:
    """Blocking single embedding function to run in threadpool"""
    model = _embedding_model_cache or get_embedding_model()
    if model is None:
        raise RuntimeError("Embedding model not loaded")
    return model.embed_single(text, instruction=instruction)

def _generate_text(prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
    """Blocking generation function to run in threadpool"""
    llm = _llm_model_cache or get_llm()
    if llm is None:
        raise RuntimeError("LLM not loaded")
    return llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Embed texts into vectors with concurrency control and batching"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    
    if not is_embedding_loaded():
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    async with semaphore_context(embed_semaphore, config.ACQUIRE_TIMEOUT):
        try:
            batch_size = config.EMBEDDING_BATCH_SIZE
            all_embeddings = []
            
            for i in range(0, len(request.texts), batch_size):
                batch = request.texts[i:i + batch_size]
                
                try:
                    batch_embeddings = await asyncio.wait_for(
                        asyncio.to_thread(_embed_batch, batch, request.instruction),
                        timeout=config.EMBED_TIMEOUT
                    )
                    all_embeddings.extend(batch_embeddings)
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=f"Embedding timeout after {config.EMBED_TIMEOUT}s"
                    )
                except Exception as e:
                    logger.error(f"Embedding batch error: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail="Internal server error during embedding"
                    )
            
            return {"embeddings": all_embeddings}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error during embedding"
            )

@app.post("/embed/single", response_model=Dict[str, Any])
async def embed_single(request: EmbedSingleRequest):
    """Embed single text into vector with concurrency control"""
    if not request.text:
        raise HTTPException(status_code=400, detail="text cannot be empty")
    
    if not is_embedding_loaded():
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    async with semaphore_context(embed_semaphore, config.ACQUIRE_TIMEOUT):
        try:
            embedding = await asyncio.wait_for(
                asyncio.to_thread(_embed_single, request.text, request.instruction),
                timeout=config.EMBED_TIMEOUT
            )
            return {"embedding": embedding}
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Embedding timeout after {config.EMBED_TIMEOUT}s"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error during embedding"
            )

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt using LLM with concurrency control"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")
    
    if not is_llm_loaded():
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    async with semaphore_context(llm_semaphore, config.ACQUIRE_TIMEOUT):
        try:
            llm = _llm_model_cache or get_llm()
            
            if hasattr(llm, '_generate_async'):
                try:
                    text = await asyncio.wait_for(
                        llm._generate_async(
                            request.prompt,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens
                        ),
                        timeout=config.LLM_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=f"Generation timeout after {config.LLM_TIMEOUT}s"
                    )
            else:
                try:
                    text = await asyncio.wait_for(
                        asyncio.to_thread(
                            _generate_text,
                            request.prompt,
                            request.temperature,
                            request.max_tokens
                        ),
                        timeout=config.LLM_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=f"Generation timeout after {config.LLM_TIMEOUT}s"
                    )
            
            return {"text": text}
            
        except HTTPException:
            raise
        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"Generation error: {error_msg}")
            
            if "memory" in error_msg.lower() or "ram" in error_msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM service error: {error_msg}. "
                           "Model requires more memory than available. "
                           "Consider using a smaller model or increasing system memory."
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM service error: {error_msg}. "
                           "Model not found. Please ensure the model is available in Ollama."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM service error: {error_msg}"
                )
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error during generation: {str(e)}"
            )

@app.get("/config")
async def get_config_endpoint():
    """Get model service configuration"""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_device": config.EMBEDDING_DEVICE,
        "embedding_batch_size": config.EMBEDDING_BATCH_SIZE,
        "embed_concurrency": config.EMBED_CONCURRENCY,
        "embed_timeout": config.EMBED_TIMEOUT,
        "llm_model": config.LLM_MODEL,
        "llm_device": config.LLM_DEVICE,
        "llm_concurrency": config.LLM_CONCURRENCY,
        "llm_timeout": config.LLM_TIMEOUT,
        "threadpool_max_workers": config.THREADPOOL_MAX_WORKERS,
        "acquire_timeout": config.ACQUIRE_TIMEOUT,
        "ollama_timeout": getattr(config, 'OLLAMA_TIMEOUT', 150.0),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
