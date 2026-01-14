"""
PDF OCR wrapper - extracts text from PDF pages using OCR engine
"""
from typing import Optional, Any
from rag.logging import logger
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

fitz: Optional[Any] = None
HAS_PYMUPDF: bool = False
HAS_OCR_ENGINE: bool = False
_ocr_engine: Optional[Any] = None

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    pass

try:
    from rag.ocr.main import RapidOCR
    HAS_OCR_ENGINE = True
except ImportError as e:
    logger.warning(f"[OCR] Failed to import OCR engine: {e}")
    HAS_OCR_ENGINE = False
except Exception as e:
    logger.warning(f"[OCR] OCR engine initialization error: {e}")
    HAS_OCR_ENGINE = False


def _get_ocr_engine():
    """Get or create OCR engine instance (singleton)"""
    global _ocr_engine
    if _ocr_engine is None and HAS_OCR_ENGINE:
        try:
            from pathlib import Path
            ocr_root = Path(__file__).resolve().parent
            models_dir = ocr_root / "models"
            
            params = {
                "Det.model_path": str(models_dir / "ch_PP-OCRv5_mobile_det.onnx"),
                "Cls.model_path": str(models_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
                "Rec.model_path": str(models_dir / "ch_PP-OCRv5_rec_mobile_infer.onnx"),
            }
            
            _ocr_engine = RapidOCR(params=params)
        except Exception as e:
            logger.error(f"[OCR] Failed to initialize OCR engine: {e}", exc_info=True)
            _ocr_engine = None
    return _ocr_engine


def extract_text_from_page(page: Any) -> str:
    """
    Extract text from PDF page using OCR engine
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        Extracted text string
    """
    if not HAS_OCR_ENGINE:
        logger.warning("[OCR] OCR engine not available")
        return ""

    if not HAS_PYMUPDF:
        logger.warning("[OCR] PyMuPDF not available - OCR fallback disabled")
        return ""

    try:
        # Convert page to high-resolution image for better OCR accuracy
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),  # 2x zoom for better OCR accuracy
            alpha=False
        )

        if not HAS_PIL:
            logger.error("[OCR] PIL not available")
            return ""

        # Convert to PIL Image then numpy array (more efficient)
        img = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples
        )
        
        # Direct conversion to numpy array (avoid intermediate copy)
        img_array = np.array(img, dtype=np.uint8)

        ocr_engine = _get_ocr_engine()
        if ocr_engine is None:
            logger.error("[OCR] OCR engine is None, cannot extract text.")
            return ""

        result = ocr_engine(img_array)
        
        if result and hasattr(result, 'txts') and result.txts:
            text_lines = [txt for txt in result.txts if txt]
            return "\n".join(text_lines)
        
        return ""
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}", exc_info=True)
        return ""


def is_available() -> bool:
    """Check if OCR is available"""
    return HAS_PYMUPDF and HAS_OCR_ENGINE and _get_ocr_engine() is not None


def get_status() -> dict:
    """Get OCR status information"""
    return {
        "available": is_available(),
        "has_pymupdf": HAS_PYMUPDF,
        "has_ocr_engine": HAS_OCR_ENGINE,
        "engine_initialized": _ocr_engine is not None
    }

