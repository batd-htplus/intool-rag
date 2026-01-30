import json
from rag.llm.semantic.base import SemanticAnalyzer
from rag.providers.ollama.llm import OllamaLLMProvider
from rag.ingest.prompts import (
    DOCUMENT_STRUCTURE_ANALYSIS_PROMPT,
)

from rag.logging import logger
from rag.helper.json import sanitize_json

class OllamaSemanticAnalyzer(SemanticAnalyzer):
    """
    Local semantic analyzer using Ollama (Qwen, Phi-3, etc.)
    """

    def __init__(self):
        self.llm = OllamaLLMProvider(model="qwen2.5:7b-instruct-q4_K_M")

    async def analyze(self, document_text: str):
        prompt = DOCUMENT_STRUCTURE_ANALYSIS_PROMPT.format(
            document_text=document_text
        )

        response = await self.llm.generate(prompt)

        logger.info((f"OllamaSemanticAnalyzer prompt: {prompt}"))
        logger.info((f"OllamaSemanticAnalyzer response: {response}"))

        if isinstance(response, dict):
            return response

        if isinstance(response, str):
            return sanitize_json(response)

        raise TypeError(f"Ollama returned unsupported type: {type(response)}")

