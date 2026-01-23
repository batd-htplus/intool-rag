"""
Data Normalization & Validation Module
======================================
Optimized data standardization for Qdrant storage:
- Text quality validation
- Metadata standardization
- Payload structure validation
- Performance optimization
"""

from typing import Dict, Any, Optional, List
import re
from pathlib import Path
from rag.logging import logger
from rag.query.prompt import clean_text_multilang

# Minimum text quality thresholds
MIN_TEXT_LENGTH = 10
MIN_PRINTABLE_RATIO = 0.7
MAX_TEXT_LENGTH = 50000  # Prevent extremely long texts

# Metadata field constraints
MAX_FILENAME_LENGTH = 255
MAX_SOURCE_LENGTH = 500
MAX_PROJECT_LENGTH = 100
MAX_LANGUAGE_LENGTH = 10

# Valid document types
VALID_DOC_TYPES = {"pdf", "docx", "xlsx", "pptx", "txt", "markdown", "plain_text"}

# Valid extractors
VALID_EXTRACTORS = {"pymupdf", "ocr", "docx", "xlsx", "pptx", "txt"}

def _calculate_printable_ratio(text: str) -> float:
    """Calculate ratio of printable characters"""
    if not text:
        return 0.0
    printable_chars = sum(1 for c in text if c.isprintable() or ord(c) > 127)
    return printable_chars / len(text)


def _is_valid_text(text: str, min_length: int = MIN_TEXT_LENGTH) -> bool:
    """
    Validate text quality before processing.
    
    Args:
        text: Text to validate
        min_length: Minimum required length
        
    Returns:
        True if text is valid for processing
    """
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    if len(text) < min_length:
        return False
    
    is_table_content = "[TABLE]" in text and "[/TABLE]" in text
    
    printable_ratio = _calculate_printable_ratio(text)
    if printable_ratio < MIN_PRINTABLE_RATIO:
        return False
    
    whitespace_ratio = sum(1 for c in text if c.isspace()) / len(text)
    if is_table_content:
        if whitespace_ratio > 0.6:
            return False
    else:
        if whitespace_ratio > 0.8:
            return False
    
    return True


def normalize_text_for_storage(text: str) -> Optional[str]:
    """
    Normalize and validate text for Qdrant storage.
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text or None if invalid
    """
    if not text:
        return None
    
    # Clean multilingual text
    cleaned = clean_text_multilang(text)
    
    if not cleaned or not cleaned.strip():
        return None
    
    # Length validation
    if len(cleaned) > MAX_TEXT_LENGTH:
        logger.warning(f"Text too long ({len(cleaned)} chars), truncating to {MAX_TEXT_LENGTH}")
        cleaned = cleaned[:MAX_TEXT_LENGTH]
    
    # Final validation
    if not _is_valid_text(cleaned):
        return None
    
    return cleaned.strip()

def normalize_filename(filename: str) -> str:
    """Normalize filename for storage"""
    if not filename:
        return "unknown"
    
    # Extract filename from path if needed
    if "/" in filename or "\\" in filename:
        filename = Path(filename).name
    
    # Truncate if too long
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = Path(filename).stem, Path(filename).suffix
        max_name_len = MAX_FILENAME_LENGTH - len(ext) - 1
        filename = name[:max_name_len] + ext
    
    return filename


def normalize_source(source: str) -> str:
    """Normalize source path for storage"""
    if not source:
        return ""
    
    # Truncate if too long
    if len(source) > MAX_SOURCE_LENGTH:
        # Keep last part of path
        source = "..." + source[-(MAX_SOURCE_LENGTH - 3):]
    
    return source


def normalize_project(project: str) -> str:
    """Normalize project name for storage"""
    if not project:
        return "default"
    
    # Remove invalid characters
    project = re.sub(r'[^\w\-_.]', '_', project)
    
    # Truncate if too long
    if len(project) > MAX_PROJECT_LENGTH:
        project = project[:MAX_PROJECT_LENGTH]
    
    return project.lower()


def normalize_language(language: str) -> str:
    """Normalize language code for storage"""
    if not language:
        return "en"
    
    # Standardize language codes
    lang_map = {
        "vi": "vi",
        "en": "en",
        "ja": "ja",
        "japan": "ja",
        "japanese": "ja",
        "vietnamese": "vi",
        "english": "en",
    }
    
    language = language.lower().strip()
    language = lang_map.get(language, language)
    
    # Truncate if too long
    if len(language) > MAX_LANGUAGE_LENGTH:
        language = language[:MAX_LANGUAGE_LENGTH]
    
    return language


def normalize_doc_type(doc_type: Optional[str]) -> str:
    """Normalize document type for storage"""
    if not doc_type:
        return "plain_text"
    
    doc_type = doc_type.lower().strip()
    
    # Map to valid types
    if doc_type not in VALID_DOC_TYPES:
        # Try to infer from common patterns
        if "markdown" in doc_type or "md" in doc_type:
            return "markdown"
        return "plain_text"
    
    return doc_type


def normalize_extractor(extractor: Optional[str]) -> str:
    """Normalize extractor name for storage"""
    if not extractor:
        return "unknown"
    
    extractor = extractor.lower().strip()
    
    if extractor not in VALID_EXTRACTORS:
        return "unknown"
    
    return extractor

def normalize_metadata(
    metadata: Dict[str, Any],
    source_filename: str,
    project: str,
    language: str,
    doc_id: str
) -> Dict[str, Any]:
    """
    Normalize and standardize metadata for Qdrant payload.
    
    Args:
        metadata: Raw metadata from document
        source_filename: Source filename
        project: Project name
        language: Language code
        doc_id: Document ID
        
    Returns:
        Normalized metadata dictionary
    """
    normalized = {}
    
    # Required fields
    normalized["project"] = normalize_project(project)
    normalized["doc_id"] = doc_id
    normalized["language"] = normalize_language(language)
    
    filename = metadata.get("filename") or source_filename
    normalized["filename"] = normalize_filename(filename)
    
    source = metadata.get("source") or source_filename
    normalized["source"] = normalize_source(source)
    
    normalized["doc_type"] = normalize_doc_type(metadata.get("doc_type"))
    
    normalized["extractor"] = normalize_extractor(metadata.get("extractor"))
    
    if "page" in metadata and metadata["page"] is not None:
        try:
            normalized["page"] = int(metadata["page"])
        except (ValueError, TypeError):
            pass
    
    if "total_pages" in metadata and metadata["total_pages"] is not None:
        try:
            normalized["total_pages"] = int(metadata["total_pages"])
        except (ValueError, TypeError):
            pass
    
    # Boolean fields
    if "is_ocr" in metadata:
        normalized["is_ocr"] = bool(metadata.get("is_ocr", False))
    
    if "has_table" in metadata:
        normalized["has_table"] = bool(metadata.get("has_table", False))
    
    if normalized.get("has_table"):
        if "table_rows" in metadata:
            try:
                normalized["table_rows"] = int(metadata["table_rows"])
            except (ValueError, TypeError):
                pass
        
        if "table_columns" in metadata:
            try:
                normalized["table_columns"] = int(metadata["table_columns"])
            except (ValueError, TypeError):
                pass
    
    # OCR metadata
    if "ocr_metadata" in metadata and isinstance(metadata["ocr_metadata"], dict):
        ocr_meta = metadata["ocr_metadata"]
        normalized_ocr = {}
        if "confidence" in ocr_meta:
            try:
                normalized_ocr["confidence"] = float(ocr_meta["confidence"])
            except (ValueError, TypeError):
                pass
        if normalized_ocr:
            normalized["ocr_metadata"] = normalized_ocr
    
    # Chunk metadata (keep minimal)
    if "chunk_meta" in metadata and isinstance(metadata["chunk_meta"], dict):
        chunk_meta = metadata["chunk_meta"]
        normalized_chunk = {}
        if "tokens" in chunk_meta:
            try:
                normalized_chunk["tokens"] = int(chunk_meta["tokens"])
            except (ValueError, TypeError):
                pass
        if "doc_type" in chunk_meta:
            normalized_chunk["doc_type"] = normalize_doc_type(chunk_meta["doc_type"])
        if normalized_chunk:
            normalized["chunk_meta"] = normalized_chunk
    
    return normalized


def validate_payload(payload: Dict[str, Any]) -> bool:
    """
    Validate payload structure before storing in Qdrant.
    
    Args:
        payload: Payload to validate
        
    Returns:
        True if payload is valid
    """
    required_fields = ["text", "project", "doc_id", "source", "language"]
    for field in required_fields:
        if field not in payload:
            logger.warning(f"Missing required field in payload: {field}")
            return False
    
    text = payload.get("text")
    if not text or not isinstance(text, str):
        logger.warning("Invalid text in payload")
        return False
    
    if not _is_valid_text(text):
        logger.warning(f"Text quality too low in payload: {len(text)} chars")
        return False
    
    if not isinstance(payload["project"], str):
        return False
    if not isinstance(payload["doc_id"], str):
        return False
    if not isinstance(payload["source"], str):
        return False
    if not isinstance(payload["language"], str):
        return False
    
    return True


def create_payload(
    text: str,
    metadata: Dict[str, Any],
    source_filename: str,
    project: str,
    language: str,
    doc_id: str,
    chunk_index: int,
    chunk_meta: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Create normalized payload for Qdrant storage.
    
    Args:
        text: Chunk text
        metadata: Document metadata
        source_filename: Source filename
        project: Project name
        language: Language code
        doc_id: Document ID
        chunk_index: Chunk index
        chunk_meta: Chunk-specific metadata
        
    Returns:
        Normalized payload or None if invalid
    """
    normalized_text = normalize_text_for_storage(text)
    if not normalized_text:
        return None
    
    normalized_metadata = normalize_metadata(
        metadata,
        source_filename,
        project,
        language,
        doc_id
    )
    
    # Add chunk-specific fields
    payload = {
        "text": normalized_text,
        "chunk_index": int(chunk_index),
        **normalized_metadata
    }
    
    # Add chunk metadata if provided
    if chunk_meta:
        payload["chunk_meta"] = chunk_meta
    
    # Validate payload
    if not validate_payload(payload):
        return None
    
    return payload
