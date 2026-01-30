from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag.routers.page_aware_v2 import router as page_aware_router

app = FastAPI(
    title="RAG Service",
    version="2.0.0",
    description="Page-aware RAG Service"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    page_aware_router,
    prefix="",
    tags=["RAG"]
)

@app.on_event("startup")
async def startup_event():
    from rag.logging import logger
    from rag.storage.faiss_index import initialize_storage
    
    logger.info("Starting up RAG Service...")
    await initialize_storage()
