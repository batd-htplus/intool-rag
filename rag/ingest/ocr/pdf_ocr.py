"""
PDF OCR wrapper - extracts text from PDF pages using OCR engine
Enhanced with DPI control, preprocessing, and confidence filtering
"""
from typing import Optional, Any, Tuple, Dict, List
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


def _preprocess_image(
    img_array: np.ndarray,
    grayscale: bool = False,
    threshold: Optional[int] = None
) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        img_array: Input image array
        grayscale: Convert to grayscale
        threshold: Apply binary threshold (0-255)
    
    Returns:
        Preprocessed image array
    """
    if not HAS_PIL:
        return img_array
    
    try:
        img = Image.fromarray(img_array)
        
        if grayscale:
            img = img.convert('L')
        
        if threshold is not None:
            img = img.point(lambda p: 255 if p > threshold else 0, mode='1')
        
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.warning(f"[OCR] Preprocessing failed: {e}")
        return img_array


def extract_text_from_page(
    page: Any,
    dpi: int = 300,
    grayscale: bool = False,
    threshold: Optional[int] = None,
    text_score: float = 0.5,
    box_thresh: float = 0.5,
    lang_hint: Optional[str] = None,
    return_confidence: bool = False
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Extract text from PDF page using OCR engine with enhanced control.
    
    Args:
        page: PyMuPDF page object
        dpi: Render DPI (default 300 for high quality)
        grayscale: Convert to grayscale before OCR
        threshold: Binary threshold (0-255), None to disable
        text_score: Minimum text confidence score (0.0-1.0)
        box_thresh: Minimum box confidence threshold (0.0-1.0)
        lang_hint: Language hint for OCR (e.g., 'vi', 'en', 'ch')
        return_confidence: Return confidence scores and metadata
    
    Returns:
        Tuple of (extracted_text, metadata_dict) if return_confidence else (extracted_text, None)
    """
    if not HAS_OCR_ENGINE:
        logger.warning("[OCR] OCR engine not available")
        return ("", None if not return_confidence else {"error": "OCR engine not available"})

    if not HAS_PYMUPDF:
        logger.warning("[OCR] PyMuPDF not available - OCR fallback disabled")
        return ("", None if not return_confidence else {"error": "PyMuPDF not available"})

    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        if not HAS_PIL:
            logger.error("[OCR] PIL not available")
            return ("", None if not return_confidence else {"error": "PIL not available"})

        img = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples
        )
        
        img_array = np.array(img, dtype=np.uint8)
        
        img_array = _preprocess_image(img_array, grayscale=grayscale, threshold=threshold)

        ocr_engine = _get_ocr_engine()
        if ocr_engine is None:
            logger.error("[OCR] OCR engine is None, cannot extract text.")
            return ("", None if not return_confidence else {"error": "OCR engine is None"})

        result = ocr_engine(
            img_array,
            text_score=text_score,
            box_thresh=box_thresh
        )
        
        if not result:
            return ("", None if not return_confidence else {"no_text": True})
        
        metadata = None
        if return_confidence:
            metadata = {
                "dpi": dpi,
                "preprocessing": {"grayscale": grayscale, "threshold": threshold},
                "text_score": text_score,
                "box_thresh": box_thresh,
                "lang_hint": lang_hint
            }
            
            if hasattr(result, 'scores') and result.scores:
                scores = result.scores
                metadata["avg_confidence"] = float(np.mean(scores)) if len(scores) > 0 else 0.0
                metadata["min_confidence"] = float(np.min(scores)) if len(scores) > 0 else 0.0
                metadata["max_confidence"] = float(np.max(scores)) if len(scores) > 0 else 0.0
                metadata["low_confidence_lines"] = sum(1 for s in scores if s < 0.5)
        
        if hasattr(result, 'to_markdown'):
            try:
                markdown_text = result.to_markdown()
                if markdown_text and markdown_text.strip():
                    return (markdown_text, metadata)
            except Exception as e:
                logger.warning(f"[OCR] to_markdown() failed, fallback to simple extraction: {e}")
        
        if hasattr(result, 'boxes') and hasattr(result, 'txts') and result.boxes is not None and result.txts:
            text = _extract_text_with_layout(result.boxes, result.txts, text_score=text_score)
            return (text, metadata)
        
        if hasattr(result, 'txts') and result.txts:
            txts = result.txts
            if hasattr(result, 'scores') and result.scores:
                filtered_txts = [
                    txt for txt, score in zip(txts, result.scores)
                    if score >= text_score
                ]
                txts = filtered_txts if filtered_txts else txts
            
            text_lines = [txt for txt in txts if txt]
            return ("\n".join(text_lines), metadata)
        
        return ("", metadata)
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}", exc_info=True)
        return ("", None if not return_confidence else {"error": str(e)})


def _extract_text_with_layout(
    boxes: np.ndarray,
    txts: tuple,
    text_score: float = 0.5,
    scores: Optional[List[float]] = None
) -> str:
    """
    Extract text preserving layout structure using box coordinates.
    Groups text on same line and preserves table structure.
    Filters by confidence if scores provided.
    """
    try:
        from rag.ocr.utils.to_markdown import ToMarkdown
        
        if scores and len(scores) == len(txts):
            filtered_boxes = []
            filtered_txts = []
            for box, txt, score in zip(boxes, txts, scores):
                if score >= text_score:
                    filtered_boxes.append(box)
                    filtered_txts.append(txt)
            
            if filtered_boxes:
                boxes = np.array(filtered_boxes)
                txts = tuple(filtered_txts)
        
        return ToMarkdown.to(boxes, txts)
    except Exception as e:
        logger.warning(f"[OCR] Layout extraction failed: {e}")
        if scores and len(scores) == len(txts):
            filtered_txts = [
                txt for txt, score in zip(txts, scores)
                if score >= text_score
            ]
            return "\n".join([txt for txt in filtered_txts if txt])
        return "\n".join([txt for txt in txts if txt])


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

