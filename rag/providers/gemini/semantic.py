import json
from google import genai

from rag.llm.semantic.base import SemanticAnalyzer
from rag.providers.base import BaseProvider, ProviderConfig
from rag.helper.json import sanitize_json
from rag.ingest.prompts import (
    DOCUMENT_STRUCTURE_ANALYSIS_PROMPT,
)

class GeminiSemanticAnalyzer(BaseProvider, SemanticAnalyzer):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = genai.Client(api_key=config.api_key)

    async def analyze(self, document_text: str) -> List[Dict]:
        prompt = DOCUMENT_STRUCTURE_ANALYSIS_PROMPT.format(
            document_text=document_text
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={
                "temperature": 0.0,
                "max_output_tokens": 16384,
            },
        )
        return json.loads(sanitize_json((response.text or "").strip()))
