"""
HuggingFace LLM adapter.
Uses transformers library for local model inference.
Supports: Phi-3, Qwen, Mistral, LLaMA, etc.
"""
import os
from typing import Optional, AsyncIterator
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_service.logging import logger
from model_service.llm.base import BaseLLM


class HuggingFaceAdapter(BaseLLM):
    """
    HuggingFace adapter for local model inference.
    
    Loads any model from HuggingFace Hub or local directory.
    Supports:
    - microsoft/Phi-3-mini-4k-instruct
    - Qwen/Qwen2.5-7B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.1
    - meta-llama/Llama-2-7b-chat-hf
    - etc.
    """
    
    def __init__(self):
        """Initialize HuggingFace adapter"""
        self.model_name = os.getenv("LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        self.device = os.getenv("LLM_DEVICE", "cpu")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        
        models_dir = Path(os.getenv("MODELS_DIR", "./models"))
        model_basename = self.model_name.split("/")[-1]
        local_model_path = models_dir / model_basename
        
        if local_model_path.exists():
            model_path = str(local_model_path)
        else:
            model_path = self.model_name
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            device_obj = torch.device(self.device)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(device_obj)
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Cannot load model {model_path}: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response using HuggingFace model"""
        try:
            temp = temperature if temperature is not None else self.temperature
            max_tok = max_tokens if max_tokens is not None else self.max_tokens
            
            device_obj = torch.device(self.device)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device_obj)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response if present
            if prompt in response:
                response = response[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"HuggingFace generation error: {str(e)}")
            raise RuntimeError(f"Failed to generate with HuggingFace: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Generate response as stream (async).
        
        Note: HuggingFace models don't natively support streaming like Ollama.
        This generates full response then yields chunks.
        For true streaming, consider using text-generation-webui or vLLM.
        """
        response = self.generate(prompt, temperature, max_tokens)
        
        # Yield in chunks for compatibility
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i+chunk_size]
    
    def is_ready(self) -> bool:
        """Check if model is ready"""
        try:
            return self.model is not None and self.tokenizer is not None
        except Exception:
            return False
    
    def get_info(self) -> dict:
        """Get HuggingFace adapter info"""
        return {
            "backend": "huggingface",
            "model": self.model_name,
            "device": self.device,
            "ready": self.is_ready()
        }
