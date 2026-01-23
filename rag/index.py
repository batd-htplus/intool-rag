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
    
    def _build_prompt(
        self,
        question: str,
        filtered_results: List,
        chat_history: Optional[list] = None
    ) -> str:
        """
        Build optimized prompt from question and filtered results.
        
        This method:
        - Formats context with metadata and relevance ordering
        - Handles empty context gracefully
        - Supports both standard and chat modes
        - Ensures consistent prompt structure
        
        Args:
            question: User question
            filtered_results: Filtered retrieval results
            chat_history: Optional conversation history
            
        Returns:
            Optimized prompt string
        """
        if not filtered_results:
            context = ""
            if chat_history:
                return build_chat_prompt(context, chat_history, question)
            else:
                return build_prompt(context, question)
        else:
            context = format_context_with_metadata(
                filtered_results,
                query=question,
                max_results=config.CONTEXT_MAX_RESULTS,
                max_text_length=config.CONTEXT_MAX_TEXT_LENGTH
            )
            
            if chat_history:
                return build_chat_prompt(context, chat_history, question)
            else:
                return build_prompt(context, question)
    
    def _is_structured_data_query(self, question: str) -> bool:
        """
        Detect if question likely requires structured data analysis.
        
        This is a heuristic to identify queries that benefit from
        structured data (tables, lists, etc.) boost.
        
        Args:
            question: User question
            
        Returns:
            True if question likely requires structured data analysis
        """
        if not question:
            return False
        
        question_lower = question.lower()
        
        # Generic keywords indicating structured data queries
        # Not domain-specific - works for invoices, reports, schedules, etc.
        structured_keywords = [
            "table", "row", "column", "list", "item",
            "calculate", "sum", "total", "breakdown", "summary",
            "compare", "difference", "amount", "quantity", "count"
        ]
        
        return any(keyword in question_lower for keyword in structured_keywords)
    
    def _apply_query_specific_boost(
        self,
        results: List,
        question: str
    ) -> List:
        """
        Apply additional boost to structured data chunks for structured data queries.
        
        This provides a second-level boost when the query type matches
        the chunk type, improving precision for structured data questions.
        
        Args:
            results: Retrieval results
            question: User question
            
        Returns:
            Results with boosted scores (re-sorted)
        """
        if not config.STRUCTURED_DATA_BOOST_MULTIPLIER or config.STRUCTURED_DATA_BOOST_MULTIPLIER <= 1.0:
            return results
        
        is_structured_query = self._is_structured_data_query(question)
        if not is_structured_query:
            return results
        
        # Apply additional boost to structured data chunks
        for r in results:
            if hasattr(r, "metadata") and r.metadata:
                metadata = r.metadata
                is_structured = (
                    metadata.get("has_table", False) or
                    metadata.get("doc_type") == "table" or
                    metadata.get("has_list", False) or
                    metadata.get("doc_type") == "list"
                )
                
                if is_structured:
                    r.score = r.score * config.STRUCTURED_DATA_BOOST_MULTIPLIER
        
        # Re-sort after boosting
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    async def query(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,  # Deterministic output - no randomness
        max_tokens: Optional[int] = None,
        chat_history: Optional[list] = None,
        include_sources: bool = True,
        include_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Execute RAG query:
        1. Retrieve relevant documents (with hybrid search)
        2. Apply query-specific boost for structured data
        3. Filter by relevance threshold
        4. Build prompt with context
        5. Generate answer
        
        Returns answer with optional sources and prompt
        
        Args:
            question: User question
            filters: Optional filters
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            chat_history: Optional conversation history
            include_sources: Whether to include sources in response
            include_prompt: Whether to include prompt in response (debug)
            
        Returns:
            Dictionary with answer, optional sources, and optional prompt
        """
        try:
            enhanced_filters = filters.copy() if filters else {}
            
            results = await retrieve(question, enhanced_filters)
            
            results = self._apply_query_specific_boost(results, question)
            
            filtered_results = self._filter_by_relevance(results)
            prompt = self._build_prompt(question, filtered_results, chat_history)
            
            llm = get_container().get_llm_provider()
            try:
                answer = await llm.generate(prompt, temperature, max_tokens)
            except LLMError:
                # Re-raise LLM errors as-is
                raise
            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                raise LLMError(f"Failed to generate answer: {str(e)}")
            
            response = {"answer": answer}
            
            if include_sources:
                response["sources"] = self._format_sources_metadata(filtered_results)
            
            if include_prompt:
                response["prompt"] = prompt
            
            return response
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            raise
    
    def _format_sources_metadata(self, results: List) -> List[Dict[str, Any]]:
        """
        Format sources with simple metadata
        """
        formatted_sources = []
        
        for r in results:
            metadata = getattr(r, "metadata", {}) or {}
            
            filename = metadata.get("filename")
            if not filename:
                source = metadata.get("source", "")
                if source:
                    filename = source.split("/")[-1] if "/" in source else source
                else:
                    filename = "unknown"
            
            text_preview = r.text[:200]
            if len(r.text) > 200:
                text_preview += "..."
            
            source_info = {
                "text": text_preview,
                "score": round(r.score, 4),
                "filename": filename,
            }
            
            formatted_sources.append(source_info)
        
        return formatted_sources
    
    async def query_stream(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,  # Deterministic output - no randomness
        max_tokens: Optional[int] = None,
        chat_history: Optional[list] = None
    ) -> AsyncIterator[str]:
        """Stream RAG query response"""
        try:
            results = await retrieve(question, filters)
            filtered_results = self._filter_by_relevance(results)
            prompt = self._build_prompt(question, filtered_results, chat_history)
            
            llm = get_container().get_llm_provider()
            try:
                async for chunk in llm.generate_stream(prompt, temperature, max_tokens):
                    yield chunk
            except LLMError:
                # Re-raise LLM errors as-is
                raise
            except Exception as e:
                logger.error(f"LLM stream generation failed: {str(e)}")
                raise LLMError(f"Failed to stream answer: {str(e)}")
        except Exception as e:
            logger.error(f"Stream query error: {str(e)}")
            raise

engine = QueryEngine()
