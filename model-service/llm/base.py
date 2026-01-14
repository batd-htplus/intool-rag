"""
Base LLM interface - all adapters must implement this.
Defines the contract that core application depends on.
"""
from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.
    
    Core application uses ONLY this interface.
    Actual implementation (Ollama, HuggingFace, etc.) is hidden behind adapter.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Generate response as stream.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Yields:
            Text chunks
            
        Raises:
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if LLM is ready to serve requests.
        
        Returns:
            True if ready, False otherwise
        """
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """
        Get LLM info (model name, backend type, etc.).
        
        Returns:
            Dict with info
        """
        pass
