"""
Ollama LLM adapter.
Uses Ollama API to interact with local LLM models.
"""
import httpx
import os
from typing import Optional, AsyncIterator
from ai.logging import logger
from ai.llm.base import BaseLLM


class OllamaAdapter(BaseLLM):
    """
    Ollama adapter for local LLM inference.
    
    Supports any model available in Ollama:
    - phi3:mini
    - phi3:medium
    - qwen2.5:7b
    - mistral
    - etc.
    """
    
    def __init__(self):
        """Initialize Ollama adapter"""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "phi3:mini"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "150.0"))
    
    def _get_client(self):
        """Get httpx client for Ollama API"""
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response using Ollama API"""
        try:
            temp = temperature if temperature is not None else self.temperature
            max_tok = max_tokens if max_tokens is not None else self.max_tokens
            
            with self._get_client() as client:
                response = client.post(
                    "/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temp,
                        "num_predict": max_tok,
                        "stream": False
                    }
                )
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", f"HTTP {response.status_code}")
                        logger.error(
                            f"Ollama API error ({response.status_code}): {error_msg}. "
                            f"Model: {self.model_name}, URL: {self.base_url}"
                        )
                        raise RuntimeError(f"Ollama error: {error_msg}")
                    except ValueError:
                        logger.error(
                            f"Ollama HTTP error {response.status_code}: {response.text[:500]}"
                        )
                        raise RuntimeError(
                            f"Ollama HTTP error {response.status_code}: {response.text[:200]}"
                        )
                
                response.raise_for_status()
                result = response.json()
                
                if "error" in result:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Ollama returned error in response: {error_msg}")
                    raise RuntimeError(f"Ollama error: {error_msg}")
                
                return result.get("response", "").strip()
        
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", str(e))
                logger.error(
                    f"Ollama HTTP error {e.response.status_code}: {error_msg}. "
                    f"Model: {self.model_name}"
                )
                raise RuntimeError(f"Ollama error: {error_msg}")
            except (ValueError, AttributeError):
                logger.error(f"Ollama HTTP error: {str(e)}")
                raise RuntimeError(f"Ollama HTTP error: {str(e)}")
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Ollama generation error: {str(e)}")
            raise RuntimeError(f"Ollama generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """Generate response as stream using Ollama API"""
        try:
            temp = temperature if temperature is not None else self.temperature
            max_tok = max_tokens if max_tokens is not None else self.max_tokens
            
            async with httpx.AsyncClient(
                base_url=self.base_url, 
                timeout=self.timeout
            ) as client:
                async with client.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temp,
                        "num_predict": max_tok,
                        "stream": True
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json
                                data = json.loads(line)
                                chunk = data.get("response", "")
                                if chunk:
                                    yield chunk
                            except Exception:
                                pass
        
        except Exception as e:
            logger.error(f"Ollama stream error: {str(e)}")
            raise
    
    def is_ready(self) -> bool:
        """Check if Ollama service is ready"""
        try:
            with self._get_client() as client:
                response = client.get("/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False
    
    def get_info(self) -> dict:
        """Get Ollama adapter info"""
        return {
            "backend": "ollama",
            "model": self.model_name,
            "base_url": self.base_url,
            "ready": self.is_ready()
        }
