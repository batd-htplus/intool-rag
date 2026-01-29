"""
Gemini Embedding Provider
=========================

Uses Google Gemini API for embeddings.

Benefits:
- Semantic consistency: Same model analyzes structure AND embeds content
- No external dependencies: Uses existing Gemini API key
- High quality: Gemini embeddings understand document context

Used in both phases:
- BUILD: Embed semantic nodes (sections/chunks after LLM structure analysis)
- QUERY: Embed user query for FAISS search

IMPORTANT: Embeds only semantic nodes/chunks, NOT full PDF text.
"""

from typing import List, Optional
from rag.logging import logger
from rag.config import config

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from rag.ingest.gemini.config import GeminiSemanticConfig
from rag.providers.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Gemini API embeddings provider.
    
    Embeds semantic nodes and queries using Google's Gemini embedding model.
    
    Advantages:
    - Same LLM that analyzes structure → semantic consistency
    - No additional API key needed (uses GOOGLE_API_KEY)
    - High-quality semantic embeddings
    - Understands document context from same model
    
    Usage:
    - BUILD: embed_batch(node_contents) → FAISS index
    - QUERY: embed_single(user_query) → FAISS search
    """
    
    def __init__(self, config: Optional[GeminiSemanticConfig] = None):
        """
        Initialize Gemini embedding provider.
        
        Args:
            config: Optional GeminiSemanticConfig
        """
        if not HAS_GEMINI:
            raise RuntimeError(
                "Gemini not installed: pip install google-genai"
            )
        
        self.config = config or GeminiSemanticConfig()
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = "gemini-embedding-001"
        self.dimension = 768

        
        logger.info(f"[EMBED] ✓ Gemini embedding provider ready (dim={self.dimension})")
    
    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Embed single text (query or node).
        
        Args:
            text: Text to embed
            instruction: Optional instruction (e.g., "Represent this for semantic search")
            
        Returns:
            Embedding vector (768 dims)
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        try:
            input_text = text.strip()
            if instruction:
                input_text = f"{instruction}\n{text.strip()}"
            
            result = self.client.models.embed_content(
                model=self.model,
                content=input_text,
            )
            
            embedding = result.embedding
            return list(embedding)
            
        except Exception as e:
            logger.error(f"[EMBED] Gemini single embedding failed: {e}")
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Embed batch of texts (node contents for FAISS indexing).
        
        Args:
            texts: List of texts to embed (semantic nodes/chunks)
            instruction: Optional instruction
            batch_size: Batch size for API calls (Gemini handles internally)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            input_texts = []
            for text in texts:
                if not text or not text.strip():
                    input_texts.append("")
                else:
                    t = text.strip()
                    if instruction:
                        t = f"{instruction}\n{t}"
                    input_texts.append(t)
            
            embeddings = []
            for text in input_texts:
                if not text:
                    embeddings.append([0.0] * self.dimension)
                else:
                    result = self.client.models.embed_content(
                        model=self.model,
                        content=text,
                    )
                    embeddings.append(list(result.embedding))
            
            logger.debug(f"[EMBED] Embedded {len(embeddings)} texts via Gemini")
            return embeddings
            
        except Exception as e:
            logger.error(f"[EMBED] Gemini batch embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension (768 for Gemini)"""
        return self.dimension


def create_gemini_embedding_provider(
    config: Optional[GeminiSemanticConfig] = None
) -> GeminiEmbeddingProvider:
    """
    Create Gemini embedding provider.
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized provider
    """
    return GeminiEmbeddingProvider(config)
