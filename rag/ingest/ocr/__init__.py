# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from .main import RapidOCR
from .utils import LoadImageError, VisRes
from .utils.typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion

from .pdf_ocr import extract_text_from_page, is_available, get_status

__all__ = [
    'RapidOCR',
    'LoadImageError',
    'VisRes',
    'EngineType',
    'LangCls',
    'LangDet',
    'LangRec',
    'ModelType',
    'OCRVersion',
    'extract_text_from_page',
    'is_available',
    'get_status',
]
