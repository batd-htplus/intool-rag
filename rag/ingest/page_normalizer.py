"""
Page-Aware Text Normalizer
==========================

Purpose:
- Clean and normalize text per page
- Remove headers/footers, fix line breaks
- Normalize unicode, remove noise
- Maintain page structure for indexing

Output:
{
  "page": 12,
  "clean_text": "..."
}
"""

from typing import Optional, Dict, Any
import re
import unicodedata
from rag.logging import logger


class TextNormalizer:
    """Normalize text for PageIndex - no page merging"""
    
    def __init__(self):
        # Common header/footer patterns
        self.header_footer_patterns = [
            r'^Page \d+\s*$',  # "Page 12"
            r'^\d+\s*$',  # Just page number
            r'^-+\s*$',  # Separator lines
            r'^\s*[\|\-]+\s*$',  # Table separators
        ]
        
        # Watermark patterns
        self.watermark_patterns = [
            r'\[DRAFT\]',
            r'\[CONFIDENTIAL\]',
            r'©.*?\d{4}',
        ]
        
        # Compile patterns
        self.header_footer_regex = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in self.header_footer_patterns
        ]
        
        self.watermark_regex = [
            re.compile(p, re.IGNORECASE)
            for p in self.watermark_patterns
        ]
    
    def normalize(self, text: str) -> str:
        """
        Normalize text on a single page.
        
        Steps:
        1. Remove watermarks
        2. Normalize unicode
        3. Fix line breaks
        4. Remove extra whitespace
        5. Clean headers/footers
        
        Args:
            text: Raw text from page
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        for pattern in self.watermark_regex:
            text = pattern.sub("", text)
        
        text = unicodedata.normalize("NFKC", text)
        
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        lines = text.split('\n')
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if line:
                fixed_lines.append(line)
        
        text = '\n'.join(fixed_lines)
        text = text.replace('\t', ' ')
        text = re.sub(r'  +', ' ', text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line is likely header/footer
            is_header_footer = any(
                pattern.match(line)
                for pattern in self.header_footer_regex
            )
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        text = text.strip()
        
        return text
    
    def is_valid_page_text(self, text: str, min_length: int = 20) -> bool:
        """
        Check if text is valid for PageIndex.
        
        Args:
            text: Text to validate
            min_length: Minimum required length
            
        Returns:
            True if valid
        """
        if not text:
            return False
        
        text = text.strip()
        if len(text) < min_length:
            return False
        
        # Check for minimum printable characters
        printable = sum(1 for c in text if c.isprintable() or ord(c) > 127)
        if printable / len(text) < 0.7:  # Less than 70% printable
            return False
        
        return True


class PageNormalizer:
    """Page-level normalization"""
    
    def __init__(self):
        self.text_normalizer = TextNormalizer()
    
    def normalize_page(self, page: int, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Normalize a single page.
        
        Args:
            page: Page number (1-based)
            raw_text: Raw text from page
            
        Returns:
            Normalized page dict or None if invalid
        """
        if page <= 0:
            logger.warning(f"Invalid page number: {page}")
            return None
        
        # Check raw text validity
        if not raw_text or not raw_text.strip():
            logger.debug(f"Page {page}: Empty content")
            return None
        
        # Normalize text
        clean_text = self.text_normalizer.normalize(raw_text)
        
        # Validate normalized text
        if not self.text_normalizer.is_valid_page_text(clean_text):
            logger.warning(f"Page {page}: Text too short or low quality after normalization")
            return None
        
        return {
            "page": page,
            "clean_text": clean_text,
        }
    
    def normalize_pages(self, pages_data: list) -> Dict[int, str]:
        """
        Normalize multiple pages.
        
        Args:
            pages_data: List of {"page": int, "raw_content": str}
            
        Returns:
            Dict of {page: clean_text}
        """
        normalized = {}
        
        for page_data in pages_data:
            page_num = page_data.get("page")
            raw_content = page_data.get("raw_content", "")
            
            result = self.normalize_page(page_num, raw_content)
            if result:
                normalized[result["page"]] = result["clean_text"]
        
        logger.info(f"Normalized {len(normalized)} pages")
        return normalized


# Convenience functions
_normalizer = None

def get_page_normalizer() -> PageNormalizer:
    """Get singleton normalizer"""
    global _normalizer
    if _normalizer is None:
        _normalizer = PageNormalizer()
    return _normalizer


def normalize_page_text(page: int, text: str) -> Optional[Dict[str, Any]]:
    """Normalize single page"""
    return get_page_normalizer().normalize_page(page, text)


def normalize_pages_text(pages_data: list) -> Dict[int, str]:
    """Normalize multiple pages"""
    return get_page_normalizer().normalize_pages(pages_data)
