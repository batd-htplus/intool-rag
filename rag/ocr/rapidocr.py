"""
Wrapper for RapidOCR module integration
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
HAS_RAPIDOCR: bool = False
_rapidocr_engine: Optional[Any] = None

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    pass

try:
    from rag.ocr import RapidOCR
    HAS_RAPIDOCR = True
except ImportError as e:
    logger.warning(f"[OCR] Failed to import RapidOCR: {e}")
    HAS_RAPIDOCR = False
except Exception as e:
    logger.warning(f"[OCR] RapidOCR initialization error: {e}")
    HAS_RAPIDOCR = False


def _get_rapidocr_engine():
    """Get or create RapidOCR engine instance (singleton)"""
    global _rapidocr_engine
    if _rapidocr_engine is None and HAS_RAPIDOCR:
        try:
            from pathlib import Path
            ocr_root = Path(__file__).resolve().parent
            models_dir = ocr_root / "models"
            
            logger.info(f"[OCR] Initializing RapidOCR engine with models from: {models_dir}")
            
            params = {
                "Det.model_path": str(models_dir / "ch_PP-OCRv5_mobile_det.onnx"),
                "Cls.model_path": str(models_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
                "Rec.model_path": str(models_dir / "ch_PP-OCRv5_rec_mobile_infer.onnx"),
            }
            
            _rapidocr_engine = RapidOCR(params=params)
            logger.info("[OCR] RapidOCR engine initialized successfully")
        except Exception as e:
            logger.error(f"[OCR] Failed to initialize RapidOCR engine: {e}", exc_info=True)
            _rapidocr_engine = None
    return _rapidocr_engine


def extract_text_from_page(page: Any) -> str:
    """
    Extract text from PDF page using RapidOCR
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        Extracted text string
    """
    if not HAS_RAPIDOCR:
        logger.warning("[OCR] RapidOCR not available")
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

        ocr_engine = _get_rapidocr_engine()
        if ocr_engine is None:
            logger.error("[OCR] RapidOCR engine is None, cannot extract text.")
            return ""

        result = ocr_engine(img_array)
        
        if result and hasattr(result, 'txts') and result.txts:
            text_lines = [txt for txt in result.txts if txt]
            return "\n".join(text_lines)
        
        return ""
    except Exception as e:
        logger.error(f"RapidOCR extraction failed: {e}", exc_info=True)
        return ""


def is_available() -> bool:
    """Check if OCR is available"""
    return HAS_PYMUPDF and HAS_RAPIDOCR and _get_rapidocr_engine() is not None


def get_status() -> dict:
    """Get OCR status information"""
    return {
        "available": is_available(),
        "has_pymupdf": HAS_PYMUPDF,
        "has_rapidocr": HAS_RAPIDOCR,
        "engine_initialized": _rapidocr_engine is not None
    }

