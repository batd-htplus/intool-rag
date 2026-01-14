import torch
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from pathlib import Path
from model_service.config import config
from model_service.logging import logger

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class BGEEmbedding:
    """BGE-M3 embedding model for multi-lingual text"""
    
    def __init__(self):
        self.device = config.EMBEDDING_DEVICE
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        models_dir = config.MODELS_DIR
        model_name = config.EMBEDDING_MODEL.split("/")[-1]  # e.g., "bge-m3"
        local_model_path = models_dir / model_name
        
        if local_model_path.exists():
            model_path = str(local_model_path)
        else:
            model_path = config.EMBEDDING_MODEL
        
        self.model = SentenceTransformer(
            model_path,
            device=self.device
        )
    
    def embed(self, texts: List[str], instruction: str = None) -> List[List[float]]:
        """
        Embed list of texts with optional instruction
        Returns vectors of shape (len(texts), 1024)
        
        Args:
            texts: List of texts to embed
            instruction: Optional instruction text to prepend (e.g., for query or passage)
        """
        try:
            if instruction:
                texts = [f"{instruction}{text}" for text in texts]
            
            embeddings = self.model.encode(
                texts,
                batch_size=config.EMBEDDING_BATCH_SIZE,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def embed_single(self, text: str, instruction: str = None) -> List[float]:
        """
        Embed single text with optional instruction
        
        Args:
            text: Text to embed
            instruction: Optional instruction text to prepend
        """
        if instruction:
            text = f"{instruction}{text}"
        
        embedding = self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

