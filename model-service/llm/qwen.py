from typing import List, AsyncIterator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from pathlib import Path
from model_service.config import config
from model_service.logging import logger

class QwenLLM:
    """Qwen2.5 LLM model"""
    
    def __init__(self):
        self.device = config.LLM_DEVICE
        
        models_dir = config.MODELS_DIR
        model_name = config.LLM_MODEL.split("/")[-1]  # e.g., "Qwen2.5-7B-Instruct"
        local_model_path = models_dir / model_name
        
        if local_model_path.exists():
            logger.info(f"Found local model at {local_model_path}, loading from local...")
            model_path = str(local_model_path)
        else:
            logger.info(f"No local model found, downloading from Hugging Face: {config.LLM_MODEL}")
            model_path = config.LLM_MODEL
        
        logger.info(f"Loading LLM model on {self.device}")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        device_obj = torch.device(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if self.device == "cuda" and torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if self.device != "cuda" or not torch.cuda.is_available():
            self.model = self.model.to(device_obj)
        
        logger.info(f"LLM model loaded: {model_path}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response from prompt"""
        try:
            temperature = temperature or config.LLM_TEMPERATURE
            max_tokens = max_tokens or config.LLM_MAX_TOKENS
            
            device_obj = torch.device(self.device)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device_obj)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if prompt in response:
                response = response[len(prompt):].strip()
            
            return response
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> AsyncIterator[str]:
        """Stream response generation"""
        response = self.generate(prompt, temperature, max_tokens)
        
        for word in response.split():
            yield word + " "

