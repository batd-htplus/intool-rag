from abc import ABC, abstractmethod
from typing import List, Dict


class SemanticAnalyzer(ABC):
    """
    Analyze document text into semantic structure.
    Output MUST be deterministic JSON-serializable structure.
    """

    @abstractmethod
    async def analyze(self, document_text: str) -> List[Dict]:
        """
        Args:
            document_text: full normalized document text

        Returns:
            List of semantic nodes:
            [
                {
                    "id": "section-1",
                    "title": "Introduction",
                    "summary": "...",
                    "content": "...",
                    "children": [...]
                }
            ]
        """
        raise NotImplementedError
