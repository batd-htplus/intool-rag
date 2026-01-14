from typing import List, Optional, Any
from rag.logging import logger

fitz: Optional[Any] = None
HAS_PYMUPDF: bool = False

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    pass

_ocr_status_cache: Optional[dict] = None
_ocr_available_cache: Optional[bool] = None

try:
    from rag.ocr.pdf_ocr import extract_text_from_page, is_available as ocr_is_available, get_status as ocr_get_status
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    def extract_text_from_page(page: Any) -> str:
        return ""
    def ocr_is_available() -> bool:
        return False
    def ocr_get_status() -> dict:
        return {"available": False}

class Document:
    def __init__(self, text: str, metadata: Optional[dict] = None):
        self.text = text
        self.metadata = metadata or {}


def _make_document(text: str, **metadata) -> Document:
    return Document(
        text=text,
        metadata=metadata
    )


def load_pdf(filepath: str) -> List[Document]:
    documents: List[Document] = []

    if HAS_PYMUPDF:
        try:
            doc = fitz.open(filepath)

            for index, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    documents.append(
                        _make_document(
                            text,
                            source=filepath,
                            page=index + 1,
                            type="pdf",
                            extractor="pymupdf"
                        )
                    )

            doc.close()

            if documents:
                return documents

        except Exception as exc:
            logger.warning(f"[PDF] PyMuPDF failed: {exc}")

    try:
        import PyPDF2
        with open(filepath, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        _make_document(
                            text,
                            source=filepath,
                            page=index + 1,
                            type="pdf",
                            extractor="pypdf2"
                        )
                    )

        if documents:
            return documents

    except Exception as exc:
        logger.warning(f"[PDF] PyPDF2 failed: {exc}")

    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        _make_document(
                            text,
                            source=filepath,
                            page=index + 1,
                            type="pdf",
                            extractor="pdfplumber"
                        )
                    )

        if documents:
            return documents

    except Exception as exc:
        logger.warning(f"[PDF] pdfplumber failed: {exc}")

    # Cache OCR status check for performance (only check once per process)
    global _ocr_status_cache, _ocr_available_cache
    if _ocr_status_cache is None:
        _ocr_status_cache = ocr_get_status()
        _ocr_available_cache = ocr_is_available()
    
    if HAS_PYMUPDF and _ocr_available_cache:
        try:
            doc = fitz.open(filepath)
            for index, page in enumerate(doc):
                try:
                    ocr_text = extract_text_from_page(page)
                    if ocr_text.strip():
                        documents.append(
                            _make_document(
                                ocr_text,
                                source=filepath,
                                page=index + 1,
                                type="pdf",
                                extractor="rapidocr"
                            )
                        )
                except Exception:
                    pass

            doc.close()

            if documents:
                return documents
        except Exception as ocr_error:
            logger.error(f"[PDF] OCR process failed: {ocr_error}", exc_info=True)

    if not HAS_PYMUPDF:
        logger.error("[PDF] OCR unavailable: PyMuPDF not installed or failed to import")
    if not _ocr_available_cache:
        logger.error("[PDF] OCR unavailable: OCR engine not installed or failed to import")
        logger.error("[PDF] Install dependencies: opencv-python, omegaconf, onnxruntime")

    raise ValueError(f"Cannot extract text from PDF: {filepath}")

def load_docx(filepath: str) -> List[Document]:
    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(filepath)
        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )

        if not text.strip():
            return []

        return [
            _make_document(
                text,
                source=filepath,
                type="docx"
            )
        ]
    except Exception as e:
        logger.error(f"[DOCX] Loading error: {e}")
        raise

def load_xlsx(filepath: str) -> List[Document]:
    try:
        import openpyxl

        documents: List[Document] = []
        workbook = openpyxl.load_workbook(filepath)

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            rows: List[str] = []

            for row in sheet.iter_rows(values_only=True):
                line = " | ".join(
                    str(cell) if cell is not None else ""
                    for cell in row
                )
                if line.strip():
                    rows.append(line)
            
            if rows:
                documents.append(
                    _make_document(
                        "\n".join(rows),
                        source=filepath,
                        sheet=sheet_name,
                        type="xlsx"
                    )
                )

        return documents
    except Exception as e:
        logger.error(f"[XLSX] Loading error: {e}")
        raise

def load_pptx(filepath: str) -> List[Document]:
    try:
        from pptx import Presentation

        documents: List[Document] = []
        presentation = Presentation(filepath)

        for index, slide in enumerate(presentation.slides):
            texts: List[str] = []

            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    value = shape.text.strip()
                    if value:
                        texts.append(value)

            if texts:
                documents.append(
                    _make_document(
                        "\n".join(texts),
                        source=filepath,
                        slide=index + 1,
                        type="pptx"
                    )
                )

        return documents
    except Exception as e:
        logger.error(f"[PPTX] Loading error: {e}")
        raise

def load_txt(filepath: str) -> List[Document]:
    text = ""

    for encoding in ("utf-8", "latin-1"):
        try:
            with open(filepath, encoding=encoding) as file:
                text = file.read()
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"[TXT] Loading error: {e}")
            raise

    if not text.strip():
        return []

    return [
        _make_document(
            text,
            source=filepath,
            type="txt"
        )
    ]

def load(filepath: str) -> List[Document]:
    extension = filepath.lower().rsplit(".", 1)[-1]
    
    if extension == "pdf":
        return load_pdf(filepath)

    if extension == "docx":
        return load_docx(filepath)

    if extension == "xlsx":
        return load_xlsx(filepath)

    if extension == "pptx":
        return load_pptx(filepath)

    if extension == "txt":
        return load_txt(filepath)

    raise ValueError(f"Unsupported file type: {extension}")
