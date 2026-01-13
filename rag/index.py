from typing import AsyncIterator, Dict, Any, Optional, List
from rag.query.retriever import retrieve
from rag.query.prompt import build_prompt, build_chat_prompt, format_context_with_metadata
from rag.config import config
from rag.logging import logger

_llm = None

def get_llm():
    """Get LLM instance (lazy loaded, singleton)"""
    global _llm
    if _llm is None:
        from rag.llm.http_client import HTTPLLM
        _llm = HTTPLLM()
    return _llm

class QueryEngine:
    """RAG query engine"""
    
    def _filter_by_relevance(self, results: List, threshold: float = None) -> List:
        """Filter results by relevance score threshold"""
        threshold = threshold or config.LLM_RELEVANCE_THRESHOLD
        filtered = [r for r in results if r.score >= threshold]
        logger.info(f"Filtered {len(results)} results to {len(filtered)} (threshold: {threshold})")
        return filtered
    
    async def query(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Execute RAG query:
        1. Retrieve relevant documents
        2. Filter by relevance threshold
        3. Build prompt with context
        4. Generate answer
        Returns answer with optional sources
        """
        try:
            logger.info(f"Query: {question}")
            
            results = await retrieve(question, filters)
            filtered_results = self._filter_by_relevance(results)
            
            if not filtered_results:
                return {
                    "answer": "I don't have relevant information to answer this question.",
                    "sources": [],
                    "relevance_threshold": config.LLM_RELEVANCE_THRESHOLD
                }
            
            context = format_context_with_metadata(filtered_results)
            
            if chat_history:
                prompt = build_chat_prompt(context, chat_history, question)
            else:
                prompt = build_prompt(context, question)
            
            llm = get_llm()
            answer = llm.generate(prompt, temperature, max_tokens)
            
            logger.info(f"Answer: {answer[:100]}...")
            
            response = {"answer": answer}
            
            if include_sources:
                response["sources"] = [
                    {
                        "text": r.text[:200],
                        "score": round(r.score, 4),
                        "metadata": r.metadata
                    }
                    for r in filtered_results
                ]
            
            return response
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            raise
    
    async def query_stream(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list] = None
    ) -> AsyncIterator[str]:
        """Stream RAG query response"""
        try:
            logger.info(f"Stream query: {question}")
            
            results = await retrieve(question, filters)
            filtered_results = self._filter_by_relevance(results)
            
            if not filtered_results:
                yield "I don't have relevant information to answer this question."
                return
            
            context = format_context_with_metadata(filtered_results)
            
            if chat_history:
                prompt = build_chat_prompt(context, chat_history, question)
            else:
                prompt = build_prompt(context, question)
            
            llm = get_llm()
            async for chunk in llm.generate_stream(prompt, temperature, max_tokens):
                yield chunk
        except Exception as e:
            logger.error(f"Stream query error: {str(e)}")
            raise

engine = QueryEngine()
