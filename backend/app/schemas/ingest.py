from pydantic import BaseModel, Field
from typing import Optional

class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to file to ingest")
    project: str = Field(..., description="Project name")
    language: Optional[str] = Field("en", description="Document language")

class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_created: int
