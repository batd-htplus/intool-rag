import aiofiles
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid
from app.core.logging import logger

UPLOAD_DIR = "/tmp/uploads"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

class FileService:
    """Handle file upload and storage"""
    
    @staticmethod
    async def save_file(file, doc_id: str) -> str:
        """Save uploaded file temporarily"""
        try:
            filename = f"{doc_id}_{file.filename}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            
            content = await file.read()
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(content)
            
            return filepath
        except Exception as e:
            logger.error(f"File save error: {str(e)}")
            raise

    @staticmethod
    def cleanup_file(filepath: str):
        """Delete temporary file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"File cleanup error: {str(e)}")

file_service = FileService()
