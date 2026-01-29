import json
from typing import Optional, Dict

from .client import get_gemini_client
from .config import GeminiSemanticConfig

class GeminiSemanticSummarizer:
    """
    Gemini-based semantic summarizer.

    Responsibility:
    - Given a section (title + text + parent prefix),
      generate:
        - summary
    - Output MUST be valid JSON
    - This class does NOT know about document structure.
    """

    def __init__(self, config: Optional[GeminiSemanticConfig] = None):
        self.config = config or GeminiSemanticConfig()
        self.client = get_gemini_client()

    async def analyze_document_structure(
        self,
        prompt: str,
    ) -> str:
        """
        Ask Gemini to analyze document structure and return JSON.
        
        Expects JSON response with format:
        {
            "sections": [
                {"title": "...", "level": "chapter|section|subsection|paragraph", "page_index": 1, "content": "..."},
                ...
            ]
        }
        
        Args:
            prompt: Prompt containing document text and analysis instructions
            
        Returns:
            JSON string from Gemini
        """
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
            },
        )
        
        return (response.text or "").strip()
