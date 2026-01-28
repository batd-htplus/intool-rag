"""Local LLM provider using in-process LLM service."""

from typing import Optional, AsyncIterator
from rag.logging import logger
from rag.core.exceptions import LLMError
from .base import LLMProvider
from rag.llm import get_llm


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using in-process LLM service."""

    def __init__(self):
        """Initialize local LLM provider"""
        try:
            self.llm = get_llm()
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            result = self.llm.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if not result or not result.strip():
                logger.warning("Empty LLM response")
                raise LLMError("LLM returned empty response")
            
            return result
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise LLMError(f"Failed to generate text: {str(e)}")

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate text from prompt with streaming"""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            async for chunk in self.llm.generate_stream(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            raise LLMError(f"Failed to generate stream: {str(e)}")

    async def close(self):
        """Close resources (local LLM doesn't need cleanup)"""
        pass
