from typing import List, Dict
from .factory import get_semantic_analyzer


async def analyze_document(text: str) -> List[Dict]:
    """
    Public API for semantic analysis.

    Used by ingest pipeline only.
    """
    analyzer = get_semantic_analyzer()
    return await analyzer.analyze(text)
