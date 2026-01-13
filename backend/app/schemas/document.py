from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class DocumentUploadRequest(BaseModel):
    project: str = Field(..., description="Project name")
    language: Optional[str] = Field("en", description="Document language: en, vi, ja")

class DocumentMetadata(BaseModel):
    id: str
    project: str
    source: str
    language: str
    uploaded_at: datetime
    status: str

class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    project: str
    status: str
    message: str
