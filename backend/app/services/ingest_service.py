from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from app.services.file_service import file_service
from app.services.rag_service import rag_service
from app.core.logging import logger

document_store: Dict[str, Dict[str, Any]] = {}

class IngestService:
    """Handle document ingestion workflow"""
    
    async def ingest_document(
        self,
        file,
        project: str,
        language: str,
        doc_id: str
    ):
        """
        Full ingestion workflow:
        1. Save file
        2. Send to RAG service
        3. Update document store
        """
        try:
            filepath = await file_service.save_file(file, doc_id)
            
            document_store[doc_id] = {
                "id": doc_id,
                "filename": file.filename,
                "project": project,
                "language": language,
                "status": "processing",
                "created_at": datetime.utcnow().isoformat(),
                "filepath": filepath
            }
            
            result = await rag_service.ingest_document(filepath, project, language)
            
            document_store[doc_id]["status"] = "completed"
            document_store[doc_id]["chunks_created"] = result.get("chunks_created", 0)
            document_store[doc_id]["completed_at"] = datetime.utcnow().isoformat()
            
            file_service.cleanup_file(filepath)
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            if doc_id in document_store:
                document_store[doc_id]["status"] = "failed"
                document_store[doc_id]["error"] = str(e)
    
    async def get_document_status(self, doc_id: str) -> Dict[str, Any]:
        """Get document ingestion status"""
        if doc_id not in document_store:
            return {
                "id": doc_id,
                "status": "not_found",
                "error": "Document not found"
            }
        
        doc = document_store[doc_id].copy()
        doc.pop("filepath", None)
        return doc
    
    async def list_documents(
        self,
        project: Optional[str] = None,
        skip: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List documents with optional filtering"""
        docs = list(document_store.values())
        
        if project:
            docs = [d for d in docs if d["project"] == project]
        
        docs.sort(key=lambda d: d["created_at"], reverse=True)
        
        result = []
        for doc in docs[skip:skip+limit]:
            d = doc.copy()
            d.pop("filepath", None)
            result.append(d)
        
        return result
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        if doc_id not in document_store:
            return None
        
        doc = document_store[doc_id].copy()
        doc.pop("filepath", None)
        return doc
    
    async def delete_document(self, doc_id: str) -> int:
        """Delete document and its vectors"""
        if doc_id not in document_store:
            return 0
        
        deleted_count = await rag_service.delete_vectors(doc_id)
        
        del document_store[doc_id]
        
        return deleted_count
    
    async def reprocess_document(self, doc_id: str) -> str:
        """Reprocess a document"""
        if doc_id not in document_store:
            return "Document not found"
        
        doc = document_store[doc_id]
        
        doc["status"] = "reprocessing"
        
        return f"Document {doc_id} marked for reprocessing"

ingest_service = IngestService()
