"""
Search Engine: Semantic Search, Load Content, Group by Page

Steps:
- Step: FAISS vector search (embed query, find top-K)
- Step: Load chunk content from chunks.json
- Step: Group chunks by page and calculate page scores
"""

from typing import List, Dict, Any, Optional
import numpy as np
from rag.agent.state import AgentState, RetrievedChunkState, PageCandidateState
from rag.agent.data_loader import AgentStorage
from rag.logging import logger
from langchain_openai import OpenAIEmbeddings


class SemanticSearcher:
    """FAISS vector search with LangChain embeddings"""
    
    def __init__(self, embeddings_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.faiss_index = None
        self.faiss_meta = {}
    
    def load_faiss(self, faiss_index_path: str, faiss_meta: Dict[str, Any]) -> None:
        """Load FAISS index"""
        try:
            import faiss
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.faiss_meta = faiss_meta
            logger.info(f"Loaded FAISS: {self.faiss_index.ntotal} vectors")
        except ImportError:
            raise RuntimeError("FAISS not installed")
    
    async def search(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """Search FAISS and return top-K results with scores"""
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not loaded")
        
        # Embed query
        query_embedding = await self.embeddings.aembed_query(query)
        query_vector = np.array([query_embedding]).astype(np.float32)
        
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:
                score = 1.0 / (1.0 + distance)  # L2 to similarity score
                results.append({"embedding_id": int(idx), "score": float(score)})
        
        return results


class ContentLoader:
    """Load chunk content from chunks.json"""
    
    def __init__(self, storage: AgentStorage):
        self.storage = storage
    
    async def load_chunks(self, search_results: List[Dict[str, Any]]) -> List[RetrievedChunkState]:
        """Load chunk content for search results"""
        
        faiss_meta = self.storage.load_faiss_meta()
        chunks = self.storage.load_chunks()
        
        embedding_to_chunk = {
            meta.embedding_id: chunk_id
            for chunk_id, meta in faiss_meta.items()
        }
        
        loaded_chunks = []
        for result in search_results:
            chunk_id = embedding_to_chunk.get(result["embedding_id"])
            if not chunk_id:
                continue
            
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue
            
            loaded_chunks.append(RetrievedChunkState(
                chunk_id=chunk_id,
                page=chunk.page_id,
                score=result["score"],
                text=chunk.text,
                metadata={"section": chunk.section, "title": chunk.title},
            ))
        
        return loaded_chunks


class PageGrouper:
    """Group chunks by page and score pages"""
    
    def group_chunks(self, chunks: List[RetrievedChunkState]) -> List[PageCandidateState]:
        """Group chunks by page with scoring"""
        
        pages = {}
        for chunk in chunks:
            if chunk.page not in pages:
                pages[chunk.page] = {"chunks": [], "total_score": 0.0}
            pages[chunk.page]["chunks"].append(chunk)
            pages[chunk.page]["total_score"] += chunk.score
        
        candidates = []
        for page_id, data in pages.items():
            chunks_list = data["chunks"]
            avg_score = data["total_score"] / len(chunks_list)
            chunk_boost = min(len(chunks_list) * 0.05, 0.15)
            combined_score = avg_score + chunk_boost
            
            candidates.append(PageCandidateState(
                page=page_id,
                score=combined_score,
                chunks=chunks_list,
                semantic_score=avg_score,
            ))
        
        candidates.sort(key=lambda p: p.score, reverse=True)
        logger.info(f"Created {len(candidates)} page candidates")
        
        return candidates


async def semantic_search(state: AgentState, searcher: SemanticSearcher) -> None:
    """Execute: Semantic Search"""
    intent_config = state.get_intent_config()
    search_results = await searcher.search(state.normalized_query, top_k=intent_config["top_k"])
    state._search_results = search_results


async def load_content(state: AgentState, loader: ContentLoader) -> None:
    """Execute: Load Chunk Content"""
    chunks = await loader.load_chunks(state._search_results)
    state.retrieved_chunks = chunks


async def group_pages(state: AgentState, grouper: PageGrouper) -> None:
    """Execute: Group Chunks by Page"""
    if state.retrieved_chunks:
        state.page_candidates = grouper.group_chunks(state.retrieved_chunks)
    else:
        state.page_candidates = []
