"""
PageIndex Builder Service
=========================

Purpose:
- Analyze page text to extract structure (chapters, sections, titles)
- Build PageIndex for entire document
- Save PageIndex for later reference and citation

This runs OFFLINE before embedding/search
"""

from typing import List, Dict, Optional, Any, Tuple
import re
from rag.logging import logger
from rag.core.page_index import PageIndex, PageIndexEntry


class StructureAnalyzer:
    """Analyze page text to identify structural elements"""
    
    def __init__(self):
        self.chapter_pattern = re.compile(
            r'^chapter\s+(\d+|[ivxlcdm]+)',
            re.IGNORECASE | re.MULTILINE
        )
        self.section_pattern = re.compile(
            r'^section\s+(\d+(?:\.\d+)*)',
            re.IGNORECASE | re.MULTILINE
        )
        self.heading_patterns = [
            (re.compile(r'^#{1,2}\s+(.+)$', re.MULTILINE), "h2"),  # Markdown H1-H2
            (re.compile(r'^([A-Z][A-Z\s]{5,})$', re.MULTILINE), "caps"),  # ALL CAPS headings
            (re.compile(r'^(\d+\.\d+\s+[A-Z].+)$', re.MULTILINE), "numbered"),  # 1.1 Heading
        ]
    
    def extract_chapter(self, text: str) -> Optional[str]:
        """Extract chapter number from text"""
        match = self.chapter_pattern.search(text)
        if match:
            return match.group(1)
        return None
    
    def extract_section(self, text: str) -> Optional[str]:
        """Extract section number from text"""
        match = self.section_pattern.search(text)
        if match:
            return match.group(1)
        return None
    
    def extract_title(self, text: str) -> Optional[str]:
        """
        Extract main title/heading from first few lines.
        
        Strategy:
        - Look for markdown headers
        - Look for ALL CAPS lines
        - Look for numbered headers
        - Return first match
        """
        lines = text.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            for pattern, style in self.heading_patterns:
                match = pattern.match(line)
                if match:
                    return match.group(1).strip()
        
        return None
    
    def analyze_page(
        self,
        page: int,
        text: str,
        prev_chapter: Optional[str] = None,
        prev_section: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Analyze page text for structure.
        
        Returns:
            (chapter, section, title)
        """
        chapter = self.extract_chapter(text) or prev_chapter
        section = self.extract_section(text) or prev_section
        title = self.extract_title(text)
        
        return chapter, section, title


class PageIndexBuilder:
    """Build PageIndex from pages"""
    
    def __init__(self):
        self.analyzer = StructureAnalyzer()
    
    def build(
        self,
        doc_id: str,
        source_filename: str,
        pages_data: List[Dict[str, Any]],
    ) -> PageIndex:
        """
        Build PageIndex from page data.
        
        Input format:
        [
            {
                "page": 1,
                "raw_content": "...",
                "clean_text": "...",
                "has_ocr": False,
                "extraction_confidence": 1.0
            }
        ]
        
        Args:
            doc_id: Document ID
            source_filename: Original filename
            pages_data: List of page dictionaries
            
        Returns:
            PageIndex object
        """
        logger.info(f"Building PageIndex for {doc_id} ({source_filename})")
        
        page_index = PageIndex(doc_id, source_filename)
        
        prev_chapter = None
        prev_section = None
        
        for page_data in pages_data:
            page = page_data.get("page")
            raw_content = page_data.get("raw_content", "")
            clean_text = page_data.get("clean_text", "")
            has_ocr = page_data.get("has_ocr", False)
            extraction_confidence = page_data.get("extraction_confidence", 1.0)
            
            analysis_text = clean_text or raw_content
            
            if not analysis_text or not analysis_text.strip():
                logger.debug(f"Page {page}: Empty, skipping")
                continue
            
            chapter, section, title = self.analyzer.analyze_page(
                page, analysis_text, prev_chapter, prev_section
            )
            
            if chapter:
                prev_chapter = chapter
            if section:
                prev_section = section
            
            entry = PageIndexEntry(
                page=page,
                raw_content=raw_content,
                clean_text=clean_text,
                chapter=chapter,
                section=section,
                subsection=None,
                title=title,
                has_ocr=has_ocr,
                extraction_confidence=extraction_confidence,
                doc_id=doc_id,
                source_filename=source_filename,
            )
            
            page_index.add_page(entry)
            
            logger.debug(
                f"Page {page}: ch={chapter}, sec={section}, title={title}"
            )
        
        total = page_index.get_page_count()
        logger.info(f"PageIndex built: {total} pages")
        
        return page_index

def build_page_index(
    doc_id: str,
    source_filename: str,
    pages_data: List[Dict[str, Any]],
) -> PageIndex:
    """
    Build PageIndex for document.
    
    Args:
        doc_id: Document ID
        source_filename: Original filename
        pages_data: List of page dictionaries
        
    Returns:
        PageIndex object
    """
    builder = PageIndexBuilder()
    return builder.build(doc_id, source_filename, pages_data)
