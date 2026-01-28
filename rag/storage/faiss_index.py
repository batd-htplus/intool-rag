"""
FAISS Vector Index Wrapper
===========================

Simple wrapper around FAISS for reading vector index.

No external database - pure local file I/O.
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from rag.logging import logger


class FAISSIndexReader:
    """Read-only FAISS index wrapper"""
    
    def __init__(self, index_path: str):
        """
        Initialize FAISS index reader.
        
        Args:
            index_path: Path to faiss.index file
        """
        if not HAS_FAISS:
            raise RuntimeError("FAISS not installed: pip install faiss-cpu")
        
        self.index_path = index_path
        self.index = None
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load FAISS index from file"""
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index: {self.index_path}")
            logger.info(f"  Dimension: {self.index.d}")
            logger.info(f"  Size: {self.index.ntotal} vectors")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector (list of floats)
            top_k: Number of results to return
            
        Returns:
            List of (embedding_id, distance) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")
        
        query_np = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(query_np, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            score = 1.0 - (dist / 2.0)
            score = max(0.0, min(1.0, score))
            results.append((int(idx), float(score)))
        
        return results
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.index is None:
            raise RuntimeError("Index not loaded")
        return self.index.d
    
    def get_size(self) -> int:
        """Get number of vectors in index"""
        if self.index is None:
            raise RuntimeError("Index not loaded")
        return self.index.ntotal


def create_faiss_index(embeddings: List[List[float]]) -> "faiss.Index":
    """
    Create FAISS index from embeddings.
    
    Used during ingest phase to build index.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        FAISS index object
    """
    if not HAS_FAISS:
        raise RuntimeError("FAISS not installed: pip install faiss-cpu")
    
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    logger.info(f"Created FAISS index: {index.ntotal} vectors, dim={index.d}")
    
    return index


def save_faiss_index(index: "faiss.Index", path: str) -> None:
    """Save FAISS index to file"""
    faiss.write_index(index, path)
    logger.info(f"Saved FAISS index to {path}")


async def search_faiss_by_vector(
    query_vector: List[float],
    limit: int = 50,
    project: Optional[str] = None,
) -> List[dict]:
    """
    Search FAISS index and return enriched results with metadata.
    
    This is the main search function for the query pipeline.
    
    Args:
        query_vector: Query embedding vector
        limit: Number of results to return
        project: Optional project filter (for future filtering)
        
    Returns:
        List of result dicts with chunk metadata
    """
    from pathlib import Path
    from rag.config import config
    from rag.storage.file_storage import FileStorageManager
    
    try:
        # Load FAISS index
        storage_path = Path(config.STORAGE_DIR)
        
        # Find the first FAISS index file (simplified - should use project filter)
        index_files = list(storage_path.glob("*_faiss.index"))
        if not index_files:
            logger.warning("No FAISS indices found")
            return []
        
        # Use the first index found (TODO: implement project-based filtering)
        index_path = str(index_files[0])
        
        reader = FAISSIndexReader(index_path)
        search_results = reader.search(query_vector, top_k=limit)
        
        # Load chunks to map FAISS IDs to chunk data
        storage = FileStorageManager(config.STORAGE_DIR)
        
        # Extract doc_id from index filename
        doc_id = index_files[0].stem.replace("_faiss", "")
        
        # Load chunks (returns dict keyed by chunk_id)
        chunks_dict = storage.load_chunks(doc_id=doc_id)
        chunks_list = list(chunks_dict.values())  # Convert to list for indexing
        
        # Enrich results with metadata
        enriched_results = []
        for faiss_id, score in search_results:
            # Find chunk by FAISS ID (or chunk index)
            if faiss_id < len(chunks_list):
                chunk = chunks_list[faiss_id]
                enriched_results.append({
                    "chunk_id": chunk.get("chunk_id", f"unknown_{faiss_id}"),
                    "text": chunk.get("text", ""),
                    "score": score,
                    "page": chunk.get("page", 0),
                    "chapter": chunk.get("metadata", {}).get("chapter"),
                    "section": chunk.get("metadata", {}).get("section"),
                    "subsection": chunk.get("metadata", {}).get("subsection"),
                    "title": chunk.get("metadata", {}).get("title"),
                    "source_filename": chunk.get("metadata", {}).get("source_filename"),
                })
        
        logger.info(f"FAISS search returned {len(enriched_results)} results")
        return enriched_results
    
    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        raise
