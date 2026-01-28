"""
Page-Based Document Loader
===========================

Purpose:
- Load PDF and extract content per page
- Handle both text-based and scanned PDFs
- Return structured page data with text/OCR content
- No processing/normalization - raw extraction only

Output format:
[
  { "page": 1, "raw_content": "..." },
  { "page": 2, "raw_content": "..." }
]
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from rag.logging import logger

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available")

try:
    from rag.ocr.pdf_ocr import extract_text_from_page, is_available as ocr_is_available
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logger.warning("OCR module not available")


@dataclass
class RawPageData:
    """Raw page data from PDF - minimal processing"""
    page: int
    raw_content: str
    has_ocr: bool = False
    extraction_confidence: float = 1.0
    
    def is_valid(self) -> bool:
        """Check if page has any content"""
        return bool(self.raw_content and self.raw_content.strip())


class PageBasedLoader:
    """Load PDF per page without aggregation"""
    
    def __init__(self):
        self.ocr_available = HAS_OCR and ocr_is_available()
    
    def load_pdf(self, filepath: str) -> List[RawPageData]:
        """
        Load PDF and extract per page.
        
        Strategy:
        1. Try text extraction first
        2. If no text on page, try OCR
        3. Return empty string if both fail
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            List of RawPageData (one per page)
        """
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF is required for PDF loading")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF not found: {filepath}")
        
        logger.info(f"Loading PDF: {filepath}")
        
        try:
            pdf_doc = fitz.open(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}")
        
        pages = []
        total_pages = pdf_doc.page_count
        
        logger.info(f"PDF has {total_pages} pages")
        
        for page_num in range(total_pages):
            try:
                page = pdf_doc[page_num]
                
                text = page.get_text(option="text")
                
                has_ocr = False
                if not text or not text.strip():
                    if self.ocr_available:
                        logger.debug(f"Page {page_num + 1}: No text found, attempting OCR")
                        text, _ = extract_text_from_page(page)
                        has_ocr = True
                        if text:
                            logger.debug(f"Page {page_num + 1}: OCR successful")
                        else:
                            logger.warning(f"Page {page_num + 1}: OCR failed, page empty")
                    else:
                        logger.warning(f"Page {page_num + 1}: No text and OCR unavailable")
                
                page_data = RawPageData(
                    page=page_num + 1,
                    raw_content=text or "",
                    has_ocr=has_ocr,
                    extraction_confidence=1.0 if not has_ocr else 0.85
                )
                
                pages.append(page_data)
                
                if page_data.is_valid():
                    logger.debug(f"Page {page_num + 1}: Loaded {len(text)} chars (OCR: {has_ocr})")
                else:
                    logger.warning(f"Page {page_num + 1}: Empty or no content")
            
            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1}: {e}")
                pages.append(RawPageData(page=page_num + 1, raw_content=""))
        
        pdf_doc.close()
        
        valid_pages = [p for p in pages if p.is_valid()]
        logger.info(f"Loaded {len(valid_pages)}/{len(pages)} pages with content")
        
        return pages
    
    def load_file(self, filepath: str) -> List[RawPageData]:
        """
        Load any supported file format.
        
        Currently supports:
        - PDF files
        
        Args:
            filepath: Path to file
            
        Returns:
            List of RawPageData
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix == ".pdf":
            return self.load_pdf(str(filepath))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")


def load_pages(filepath: str) -> List[RawPageData]:
    """
    Convenience function to load pages from file.
    
    Args:
        filepath: Path to file
        
    Returns:
        List of RawPageData (one per page)
    """
    loader = PageBasedLoader()
    return loader.load_file(filepath)
