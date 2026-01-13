from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional
import uuid
from datetime import datetime
from app.services.ingest_service import IngestService
from app.schemas.document import DocumentUploadResponse
from app.core.logging import logger

router = APIRouter(tags=["documents"])
ingest_service = IngestService()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    project: str = Form(...),
    language: Optional[str] = Form("en"),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and ingest a document.
    Supports: PDF, DOCX, XLSX
    Languages: en, vi, ja
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is required")
        
        doc_id = str(uuid.uuid4())
        
        if background_tasks:
            background_tasks.add_task(
                ingest_service.ingest_document,
                file,
                project,
                language,
                doc_id
            )
        else:
            await ingest_service.ingest_document(file, project, language, doc_id)
        
        return DocumentUploadResponse(
            id=doc_id,
            filename=file.filename,
            project=project,
            status="processing",
            message="Document received and queued for processing"
        )
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{doc_id}")
async def get_document_status(doc_id: str):
    """Get document ingestion status"""
    try:
        status = await ingest_service.get_document_status(doc_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
