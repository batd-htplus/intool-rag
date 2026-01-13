from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.services.ingest_service import IngestService
from app.core.logging import logger

router = APIRouter(tags=["documents"])
ingest_service = IngestService()

@router.get("/")
async def list_documents(
    project: Optional[str] = Query(None, description="Filter by project"),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """List all documents with optional filtering"""
    try:
        documents = await ingest_service.list_documents(project, skip, limit)
        return {
            "documents": documents,
            "total": len(documents),
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{doc_id}")
async def get_document(doc_id: str):
    """Get document metadata"""
    try:
        doc = await ingest_service.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except Exception as e:
        logger.error(f"Get document error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document and its vectors"""
    try:
        result = await ingest_service.delete_document(doc_id)
        return {
            "status": "deleted",
            "doc_id": doc_id,
            "vectors_deleted": result
        }
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{doc_id}/reprocess")
async def reprocess_document(doc_id: str):
    """Reprocess a document"""
    try:
        result = await ingest_service.reprocess_document(doc_id)
        return {
            "status": "reprocessing",
            "doc_id": doc_id,
            "message": result
        }
    except Exception as e:
        logger.error(f"Reprocess document error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
