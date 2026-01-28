"""
Storage module __init__

Exports file storage and FAISS index utilities
"""

from rag.storage.file_storage import FileStorageManager
from rag.storage.faiss_index import FAISSIndexReader, create_faiss_index, save_faiss_index

__all__ = [
    "FileStorageManager",
    "FAISSIndexReader",
    "create_faiss_index",
    "save_faiss_index",
]
