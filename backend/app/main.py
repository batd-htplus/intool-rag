from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api import health
from app.api.v1 import chat, documents, ingest, config
from app.core.config import settings
from app.core.logging import logger

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

admin_web_path = Path(__file__).parent.parent / "admin-web"
if admin_web_path.exists():
    app.mount("/admin/static", StaticFiles(directory=str(admin_web_path)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, prefix="/v1/chat", tags=["chat"])
app.include_router(documents.router, prefix="/v1/documents", tags=["documents"])
app.include_router(ingest.router, prefix="/v1/ingest", tags=["ingest"])
app.include_router(config.router, prefix="/v1/config", tags=["config"])

@app.get("/admin")
@app.get("/admin/")
async def admin_index():
    """Admin web interface"""
    admin_web_path = Path(__file__).parent.parent / "admin-web" / "index.html"
    if admin_web_path.exists():
        return FileResponse(str(admin_web_path))
    return {"message": "Admin web interface not found"}

@app.get("/admin/documents")
async def admin_documents():
    """Admin documents page"""
    admin_web_path = Path(__file__).parent.parent / "admin-web" / "index.html"
    if admin_web_path.exists():
        return FileResponse(str(admin_web_path))
    return {"message": "Admin web interface not found"}

@app.get("/admin/models")
async def admin_models():
    """Admin models page"""
    admin_web_path = Path(__file__).parent.parent / "admin-web" / "index.html"
    if admin_web_path.exists():
        return FileResponse(str(admin_web_path))
    return {"message": "Admin web interface not found"}

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_NAME}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Stopping {settings.APP_NAME}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )