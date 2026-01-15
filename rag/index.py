from typing import AsyncIterator, Dict, Any, Optional, List
from rag.query.retriever import retrieve
from rag.query.prompt import build_prompt, build_chat_prompt, format_context_with_metadata
from rag.config import config
from rag.logging import logger
from rag.core.container import get_container
from rag.core.exceptions import LLMError

class QueryEngine:
    """RAG query engine"""
    
    def _filter_by_relevance(self, results: List, threshold: float = None) -> List:
        """Filter results by relevance score threshold"""
        threshold = threshold or config.LLM_RELEVANCE_THRESHOLD
        filtered = [r for r in results if r.score >= threshold]
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
        1. Retrieve relevant documents (with hybrid search)
        2. Filter by relevance threshold
        3. Build prompt with context
        4. Generate answer
        Returns answer with optional sources
        """
        try:
            results = await retrieve(question, filters)
            filtered_results = self._filter_by_relevance(results)
            
            if not filtered_results:
                context = ""
                if chat_history:
                    prompt = build_chat_prompt(context, chat_history, question)
                else:
                    prompt = f"""Answer the following question. If you don't know the answer, you can say so.

QUESTION: {question}

ANSWER:"""
            else:
                context = format_context_with_metadata(filtered_results, query=question)
                
                if chat_history:
                    prompt = build_chat_prompt(context, chat_history, question)
                else:
                    prompt = build_prompt(context, question)
            
            llm = get_container().get_llm_provider()
            try:
                answer = await llm.generate(prompt, temperature, max_tokens)
            except RuntimeError as e:
                logger.error(f"LLM generation failed: {str(e)}")
                raise LLMError(str(e))
            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                raise LLMError(f"Failed to generate answer: {str(e)}")
            
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
            results = await retrieve(question, filters)
            filtered_results = self._filter_by_relevance(results)
            
            if not filtered_results:
                context = ""
                if chat_history:
                    prompt = build_chat_prompt(context, chat_history, question)
                else:
                    prompt = f"""Answer the following question. If you don't know the answer, you can say so.

QUESTION: {question}

ANSWER:"""
            else:
                context = format_context_with_metadata(filtered_results, query=question)
                
                if chat_history:
                    prompt = build_chat_prompt(context, chat_history, question)
                else:
                    prompt = build_prompt(context, question)
            
            llm = get_container().get_llm_provider()
            try:
                async for chunk in llm.generate_stream(prompt, temperature, max_tokens):
                    yield chunk
            except Exception as e:
                logger.error(f"LLM stream generation failed: {str(e)}")
                raise LLMError(f"Failed to stream answer: {str(e)}")
        except Exception as e:
            logger.error(f"Stream query error: {str(e)}")
            raise

engine = QueryEngine()
