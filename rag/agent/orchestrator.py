"""
RAG Page-Aware Agent — Main Orchestrator

Sequence:

Query → Normalize → Intent → Search → Load → Group → Select → Assemble → Generate → Validate → Format → Output

🧠 Core Principle:
Agent FINDS THE RIGHT PAGE FIRST, then answers.
NOT: Find answer first, then justify with page.
"""

import time
from typing import Dict, Any
from rag.agent.state import AgentState, QueryIntent
from rag.agent.data_loader import AgentStorage
from rag.agent.query_processor import QueryNormalizer, IntentClassifier
from rag.agent.search_engine import SemanticSearcher, ContentLoader, PageGrouper, \
    semantic_search, load_content, group_pages
from rag.agent.page_selector import PageSelector, ContextAssembler, \
    select_page, assemble_context
from rag.agent.answer_generator import PromptBuilder, AnswerGenerator, AnswerValidator, \
    ResponseFormatter, generate_answer, validate_answer, format_response
from rag.logging import logger


class PageAwareAgent:
    """
    RAG Page-Aware Agent — pipeline.
    
    🎯 Design Principles:
    1️⃣ LangChain ONLY for: embeddings, prompts, LLM calls
    2️⃣ Agent logic: page index reasoning, selection, grouping
    3️⃣ Zero complex chains: each step is simple, transparent
    4️⃣ Page is authority: PageIndex > FAISS > chunk similarity
    5️⃣ No answer without page: RULE 1 enforced strictly
    
    Architecture:
    - Storage: Reads 3 files (page_index.json, chunks.json, faiss_meta.json)
    - Searcher: FAISS wrapper (LangChain embeddings)
    - Agent Logic: Custom reasoning (no LangChain)
    - LLM: LangChain ChatOpenAI
    """
    
    def __init__(
        self,
        data_dir: str,
        faiss_index_path: str,
        llm_model: str = "gpt-4-turbo",
        embeddings_model: str = "text-embedding-3-small",
    ):
        """
        Initialize agent.
        
        Args:
            data_dir: Directory with page_index.json, chunks.json, faiss_meta.json
            faiss_index_path: Path to faiss.index file
            llm_model: LangChain LLM model
            embeddings_model: LangChain embeddings model
        """
        self.data_dir = data_dir
        
        # Initialize components
        self.storage = AgentStorage(data_dir)
        
        # Verify files exist
        if not self.storage.verify():
            raise RuntimeError("Storage verification failed - check BUILD STRUCTURE phase output")
        
        # Initialize searcher (FAISS + LangChain embeddings)
        self.searcher = SemanticSearcher(embeddings_model=embeddings_model)
        
        # Load FAISS index
        faiss_meta = self.storage.load_faiss_meta()
        self.searcher.load_faiss(faiss_index_path, faiss_meta)
        
        self.normalizer = QueryNormalizer()
        self.intent_classifier = IntentClassifier()
        self.content_loader = ContentLoader(self.storage)
        self.page_grouper = PageGrouper()
        self.page_selector = PageSelector(self.storage)
        self.context_assembler = ContextAssembler(self.storage)
        self.prompt_builder = PromptBuilder()
        self.answer_generator = AnswerGenerator(model=llm_model)
        self.answer_validator = AnswerValidator()
        self.response_formatter = ResponseFormatter()
        
        logger.info("✓ PageAwareAgent initialized")
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Execute complete 11-step query pipeline.
        
        Args:
            question: User question
            
        Returns:
            {
              "answer": "...",
              "source": {
                "page": 12,
                "chapter": "...",
                "section": "...",
                "title": "..."
              }
            }
        """
        start_time = time.time()
        
        try:
            state = AgentState()
            state.query = question
            
            logger.info(f"\n{'='*60}")
            logger.info(f"QUERY: {question}")
            logger.info(f"{'='*60}\n")
            
            await self.normalizer.normalize(state)
            
            await self.intent_classifier.classify_intent(state)
            
            await semantic_search(state, self.searcher)
            
            await load_content(state, self.content_loader, self.searcher)
            
            await group_pages(state, self.page_grouper)
            
            await select_page(state, self.page_selector)
            
            await assemble_context(state, self.context_assembler)
            
            await generate_answer(state, self.prompt_builder, self.answer_generator)
            
            await validate_answer(state, self.answer_validator)
            
            response = await format_response(state, self.response_formatter)
            
            elapsed = time.time() - start_time
            response["metadata"] = {
                "execution_time_ms": round(elapsed * 1000, 2),
                "state_summary": state.to_dict(),
            }
            
            logger.info(f"\n✓ Complete in {elapsed:.2f}s")
            logger.info(f"Response: {response['answer'][:100]}...\n")
            
            return response
        
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            return {
                "answer": f"An error occurred: {str(e)}",
                "source": None,
                "error": str(e),
            }

async def query_agent(
    question: str,
    data_dir: str,
    faiss_index_path: str,
) -> Dict[str, Any]:
    """
    Quick query with agent.
    
    Args:
        question: User question
        data_dir: Path to data files
        faiss_index_path: Path to faiss.index
        
    Returns:
        Agent response
    """
    agent = PageAwareAgent(data_dir, faiss_index_path)
    return await agent.query(question)
