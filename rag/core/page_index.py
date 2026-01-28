"""
PageIndex - Structure Understanding Module
===========================================

Purpose:
- Understand document structure at page level (chapters, sections, subsections, titles)
- Create traceable source citations
- Enable page-aware retrieval and ranking

Core Concept:
- PageIndex is the "structural memory" of the document
- Each page knows its hierarchical position (chapter, section, subsection, title)
- Used for building context and generating accurate citations
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from rag.logging import logger


class StructureLevel(Enum):
    """Hierarchy levels for document structure"""
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    TITLE = "title"
    PARAGRAPH = "paragraph"


@dataclass
class PageMetadata:
    """Metadata for a single page - the core of PageIndex"""
    page: int
    clean_text: str
    
    # Structural hierarchy
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    title: Optional[str] = None
    
    # Document context
    doc_id: Optional[str] = None
    source_filename: Optional[str] = None
    language: Optional[str] = None
    
    # Processing metadata
    has_ocr: bool = False
    extraction_confidence: float = 1.0
    processing_timestamp: Optional[str] = None
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageMetadata":
        """Create from dictionary"""
        # Extract custom_metadata if present, handle nested dict
        custom_metadata = data.pop("custom_metadata", {})
        if isinstance(custom_metadata, dict):
            data["custom_metadata"] = custom_metadata
        return cls(**data)

    def get_full_hierarchy(self) -> str:
        """Get full structural hierarchy as string for context"""
        parts = []
        if self.chapter:
            parts.append(f"Chapter {self.chapter}")
        if self.section:
            parts.append(f"Section {self.section}")
        if self.subsection:
            parts.append(f"Subsection {self.subsection}")
        if self.title:
            parts.append(f"Title: {self.title}")
        
        return " → ".join(parts) if parts else ""

    def to_citation_source(self) -> Dict[str, Any]:
        """Convert to citation source format for LLM response"""
        return {
            "page": self.page,
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "title": self.title,
            "source_file": self.source_filename,
        }


@dataclass
class PageIndexEntry:
    """Complete PageIndex entry - metadata + raw content"""
    page: int
    raw_content: str  # Original content before normalization
    clean_text: str   # Normalized text
    
    # Structural information
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    title: Optional[str] = None
    
    # Processing flags
    has_ocr: bool = False
    extraction_confidence: float = 1.0
    
    # Source tracking
    doc_id: Optional[str] = None
    source_filename: Optional[str] = None
    
    def to_page_metadata(self) -> PageMetadata:
        """Convert to PageMetadata for chunk processing"""
        return PageMetadata(
            page=self.page,
            clean_text=self.clean_text,
            chapter=self.chapter,
            section=self.section,
            subsection=self.subsection,
            title=self.title,
            doc_id=self.doc_id,
            source_filename=self.source_filename,
            has_ocr=self.has_ocr,
            extraction_confidence=self.extraction_confidence,
        )


class PageIndex:
    """
    PageIndex - Document structure understanding
    
    Responsibility:
    - Store page-level structure information
    - Provide lookup by page number
    - Support structure-aware retrieval
    """
    
    def __init__(self, doc_id: str, source_filename: str):
        self.doc_id = doc_id
        self.source_filename = source_filename
        self.pages: Dict[int, PageIndexEntry] = {}
        self._page_count = 0
    
    def add_page(self, entry: PageIndexEntry) -> None:
        """Add page to index"""
        if entry.page <= 0:
            raise ValueError(f"Invalid page number: {entry.page}")
        
        if entry.page in self.pages:
            logger.warning(f"Page {entry.page} already exists, overwriting")
        
        self.pages[entry.page] = entry
        self._page_count = max(self._page_count, entry.page)
    
    def get_page(self, page: int) -> Optional[PageIndexEntry]:
        """Get page by number"""
        return self.pages.get(page)
    
    def get_pages_by_section(self, section: str) -> List[PageIndexEntry]:
        """Get all pages in a section"""
        return [
            p for p in self.pages.values()
            if p.section == section
        ]
    
    def get_pages_by_chapter(self, chapter: str) -> List[PageIndexEntry]:
        """Get all pages in a chapter"""
        return [
            p for p in self.pages.values()
            if p.chapter == chapter
        ]
    
    def get_all_pages(self) -> List[PageIndexEntry]:
        """Get all pages in order"""
        return sorted(self.pages.values(), key=lambda p: p.page)
    
    def get_page_count(self) -> int:
        """Get total page count"""
        return self._page_count
    
    def to_dict(self, include_text: bool = False) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Args:
            include_text: If True, include raw_content and clean_text.
                          If False (default), output structure-only for page_index.json.
        """
        pages_data = {}
        for page_num, entry in self.pages.items():
            entry_dict = asdict(entry)
            if not include_text:
                # Remove text content for structure-only serialization
                entry_dict.pop("raw_content", None)
                entry_dict.pop("clean_text", None)
            pages_data[str(page_num)] = entry_dict
        
        return {
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_count": self._page_count,
            "pages": pages_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageIndex":
        """Deserialize from dictionary"""
        index = cls(data["doc_id"], data["source_filename"])
        
        for page_num_str, entry_data in data.get("pages", {}).items():
            page_num = int(page_num_str)
            entry = PageIndexEntry(
                page=page_num,
                raw_content=entry_data.get("raw_content", ""),
                clean_text=entry_data.get("clean_text", ""),
                chapter=entry_data.get("chapter"),
                section=entry_data.get("section"),
                subsection=entry_data.get("subsection"),
                title=entry_data.get("title"),
                has_ocr=entry_data.get("has_ocr", False),
                extraction_confidence=entry_data.get("extraction_confidence", 1.0),
                doc_id=entry_data.get("doc_id"),
                source_filename=entry_data.get("source_filename"),
            )
            index.add_page(entry)
        
        return index
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PageIndex":
        """Deserialize from JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save PageIndex to file"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        logger.info(f"PageIndex saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "PageIndex":
        """Load PageIndex from file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
