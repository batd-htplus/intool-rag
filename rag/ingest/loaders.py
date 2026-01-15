from typing import List, Optional, Any
import re
import string

# =====================================================
# Optional dependencies
# =====================================================

fitz: Optional[Any] = None
HAS_PYMUPDF = False

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    pass

try:
    from rag.ocr.pdf_ocr import (
        extract_text_from_page,
        is_available as ocr_is_available,
        get_status as ocr_get_status
    )
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

    def extract_text_from_page(page: Any) -> str:
        return ""

    def ocr_is_available() -> bool:
        return False

    def ocr_get_status() -> dict:
        return {"available": False}


# =====================================================
# Core model
# =====================================================

class Document:
    def __init__(self, text: str, metadata: Optional[dict] = None):
        self.text = text
        self.metadata = metadata or {}


def _make_document(text: str, **metadata) -> Document:
    return Document(text=text, metadata=metadata)


# =====================================================
# Text quality helpers
# =====================================================

_PRINTABLE = set(string.printable)

def _normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_pdf_with_layout(page) -> str:
    """
    Extract PDF text with layout preservation using PyMuPDF blocks.
    Detects and preserves table structure for accurate data extraction.
    """
    try:
        blocks = page.get_text("blocks")
        if not blocks:
            return ""
        
        text_blocks = []
        for block in blocks:
            if len(block) < 5:
                continue
            text = block[4].strip()
            if not text:
                continue
            
            x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
            center_y = (y0 + y1) / 2
            center_x = (x0 + x1) / 2
            
            text_blocks.append({
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "center_x": center_x,
                "center_y": center_y,
                "width": x1 - x0,
                "height": y1 - y0
            })
        
        if not text_blocks:
            return ""
        
        text_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
        
        lines = []
        current_line = []
        current_y = None
        line_tolerance = 5
        
        for block in text_blocks:
            y_pos = block["center_y"]
            
            if current_y is None or abs(y_pos - current_y) <= line_tolerance:
                current_line.append(block)
                if current_y is None:
                    current_y = y_pos
                else:
                    current_y = (current_y + y_pos) / 2
            else:
                if current_line:
                    line_text = _format_line_with_table_structure(current_line)
                    if line_text:
                        lines.append(line_text)
                
                current_line = [block]
                current_y = y_pos
        
        if current_line:
            line_text = _format_line_with_table_structure(current_line)
            if line_text:
                lines.append(line_text)
        
        result = "\n".join(lines)
        
        if len(result.strip()) < 50:
            dict_text = page.get_text("dict")
            if dict_text and "blocks" in dict_text:
                lines = []
                for block in dict_text["blocks"]:
                    if "lines" in block:
                        line_parts = []
                        for line in block["lines"]:
                            if "spans" in line:
                                span_texts = [span.get("text", "").strip() for span in line["spans"]]
                                line_parts.append(" ".join(span_texts))
                        if line_parts:
                            lines.append(" | ".join(line_parts))
                result = "\n".join(lines)
        
        return result.strip()
    
    except Exception:
        return ""


def _format_line_with_table_structure(blocks: list) -> str:
    """
    Format a line of text blocks, detecting and preserving table structure.
    Uses proper spacing/separators based on column alignment.
    """
    if not blocks:
        return ""
    
    if len(blocks) == 1:
        return blocks[0]["text"]
    
    blocks.sort(key=lambda b: b["x0"])
    
    gaps = []
    for i in range(len(blocks) - 1):
        gap = blocks[i + 1]["x0"] - blocks[i]["x1"]
        if gap > 5:
            gaps.append(gap)
    
    is_table_row = False
    
    if len(gaps) >= 1:
        avg_gap = sum(gaps) / len(gaps)
        if len(gaps) > 1:
            gap_variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
            gap_std = gap_variance ** 0.5
            if avg_gap > 10 and gap_std / max(avg_gap, 1) < 0.6:
                is_table_row = True
        else:
            if avg_gap > 20:
                is_table_row = True
    
    if not is_table_row and len(blocks) >= 2:
        y_positions = [b["center_y"] for b in blocks]
        y_avg = sum(y_positions) / len(y_positions)
        y_variance = sum((y - y_avg) ** 2 for y in y_positions) / len(y_positions)
        
        if y_variance < 15:
            x_positions = [b["x0"] for b in blocks]
            x_spread = max(x_positions) - min(x_positions)
            if x_spread > 80:
                is_table_row = True
    
    if not is_table_row and len(blocks) >= 3:
        text_line = " ".join([b["text"] for b in blocks])
        has_numbers = any(re.search(r'\d+', b["text"]) for b in blocks)
        has_currency = any(re.search(r'\$|USD|EUR', b["text"]) for b in blocks)
        has_table_keywords = any(re.search(r'item|quantity|rate|amount|total|subtotal|discount|shipping', b["text"], re.I) for b in blocks)
        
        if (has_numbers or has_currency or has_table_keywords) and len(blocks) >= 3:
            is_table_row = True
    
    if is_table_row:
        parts = [b["text"] for b in blocks]
        return " | ".join(parts)
    else:
        parts = []
        for i, block in enumerate(blocks):
            if i > 0:
                prev_block = blocks[i - 1]
                gap = block["x0"] - prev_block["x1"]
                if gap > 20:
                    parts.append(" | ")
                else:
                    parts.append(" ")
            parts.append(block["text"])
        return "".join(parts)


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for c in text if c in _PRINTABLE or ord(c) > 127)
    return printable / len(text)


def _is_usable_text(text: str, min_len: int = 100) -> bool:
    text = text.strip()
    if len(text) < min_len:
        return False
    if _printable_ratio(text) < 0.6:
        return False
    return True


# =====================================================
# PDF Loader
# =====================================================

def load_pdf(filepath: str) -> List[Document]:
    documents: List[Document] = []

    # ---------- 1. PyMuPDF with layout preservation (preferred)
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(filepath)

            for idx, page in enumerate(doc):
                # Try layout-preserving extraction first (better for tables/invoices)
                text = _extract_pdf_with_layout(page)
                
                # Fallback to simple text extraction if layout extraction fails
                if not text or len(text.strip()) < 50:
                    raw = page.get_text("text")
                    text = _normalize_text(raw)
                else:
                    # Layout extraction already preserves structure, only light cleanup
                    # Don't normalize newlines - they preserve table structure
                    text = re.sub(r'[ \t]+', ' ', text)  # Only normalize spaces/tabs
                    text = text.strip()

                if _is_usable_text(text, min_len=50):
                    documents.append(
                        _make_document(
                            text,
                            source=filepath,
                            page=idx + 1,
                            doc_type="pdf",
                            extractor="pymupdf",
                            is_ocr=False
                        )
                    )

            doc.close()

            if documents:
                return documents

        except Exception:
            pass

    # ---------- 2. OCR fallback (only if really needed)
    if HAS_PYMUPDF and HAS_OCR and ocr_is_available():
        try:
            doc = fitz.open(filepath)

            for idx, page in enumerate(doc):
                try:
                    # OCR already preserves layout structure (tables, key-value pairs)
                    # Don't normalize too aggressively to keep structure
                    ocr_text = extract_text_from_page(page)
                    if not ocr_text or not ocr_text.strip():
                        continue
                    
                    # Light normalization: only fix excessive whitespace, preserve newlines
                    ocr_text = re.sub(r'[ \t]+', ' ', ocr_text)  # Normalize spaces/tabs
                    ocr_text = ocr_text.strip()
                    
                    if _is_usable_text(ocr_text, min_len=50):
                        documents.append(
                            _make_document(
                                ocr_text,
                                source=filepath,
                                page=idx + 1,
                                doc_type="pdf",
                                extractor="ocr",
                                is_ocr=True
                            )
                        )
                except Exception:
                    continue

            doc.close()

            if documents:
                return documents

        except Exception:
            pass

    raise ValueError(f"[PDF] Cannot extract usable text: {filepath}")


# =====================================================
# DOCX
# =====================================================

def load_docx(filepath: str) -> List[Document]:
    from docx import Document as Docx

    doc = Docx(filepath)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    text = _normalize_text(text)

    if not _is_usable_text(text):
        return []

    return [_make_document(text, source=filepath, doc_type="docx")]


# =====================================================
# XLSX
# =====================================================

def load_xlsx(filepath: str) -> List[Document]:
    import openpyxl

    documents: List[Document] = []
    wb = openpyxl.load_workbook(filepath, data_only=True)

    for sheet in wb.sheetnames:
        ws = wb[sheet]
        rows = []

        for row in ws.iter_rows(values_only=True):
            line = " | ".join(str(c) for c in row if c is not None)
            if line.strip():
                rows.append(line)

        text = _normalize_text("\n".join(rows))
        if _is_usable_text(text):
            documents.append(
                _make_document(
                    text,
                    source=filepath,
                    sheet=sheet,
                    doc_type="xlsx"
                )
            )

    return documents


# =====================================================
# PPTX
# =====================================================

def load_pptx(filepath: str) -> List[Document]:
    from pptx import Presentation

    documents: List[Document] = []
    pres = Presentation(filepath)

    for idx, slide in enumerate(pres.slides):
        texts = []

        for shape in slide.shapes:
            if hasattr(shape, "text"):
                value = shape.text.strip()
                if value:
                    texts.append(value)

        text = _normalize_text("\n".join(texts))
        if _is_usable_text(text, min_len=50):
            documents.append(
                _make_document(
                    text,
                    source=filepath,
                    slide=idx + 1,
                    doc_type="pptx"
                )
            )

    return documents


# =====================================================
# TXT
# =====================================================

def load_txt(filepath: str) -> List[Document]:
    text = ""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            with open(filepath, encoding=enc) as f:
                text = _normalize_text(f.read())
            break
        except UnicodeDecodeError:
            continue

    if not text or not _is_usable_text(text):
        return []

    return [_make_document(text, source=filepath, doc_type="txt")]


# =====================================================
# Router
# =====================================================

def load(filepath: str) -> List[Document]:
    ext = filepath.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        return load_pdf(filepath)
    if ext == "docx":
        return load_docx(filepath)
    if ext == "xlsx":
        return load_xlsx(filepath)
    if ext == "pptx":
        return load_pptx(filepath)
    if ext == "txt":
        return load_txt(filepath)

    raise ValueError(f"Unsupported file type: {ext}")
