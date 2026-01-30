from google import genai

from rag.llm.base import LLM
from rag.providers.base import BaseProvider, ProviderConfig


class GeminiLLM(BaseProvider, LLM):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = genai.Client(api_key=config.api_key)

    async def generate(
        self, 
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        config = {}
        if temperature: config["temperature"] = temperature
        if max_tokens: config["max_output_tokens"] = max_tokens
        
        resp = self.client.models.generate_content(
            model=self.config.model or "gemini-2.5-pro",
            contents=prompt,
            config=config if config else None
        )
        return (resp.text or "").strip()

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        config = {}
        if temperature: config["temperature"] = temperature
        if max_tokens: config["max_output_tokens"] = max_tokens

        resp = self.client.models.generate_content_stream(
            model=self.config.model or "gemini-2.5-pro",
            contents=prompt,
            config=config if config else None
        )
        for chunk in resp:
            if chunk.text:
                yield chunk.text

    def is_ready(self) -> bool:
        return self.client is not None

    def get_info(self) -> dict:
        return {
            "provider": "gemini",
            "model": self.config.model,
        }
