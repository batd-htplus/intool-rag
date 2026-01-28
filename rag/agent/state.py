"""
RAG Page-Aware Agent — State Management
========================================

Maintains internal state throughout the query execution pipeline.

Agent State Flow:
Step 1 → state.query
Step 2 → state.normalized_query
Step 3 → state.intent
Step 4-5 → state.retrieved_chunks
Step 6 → state.page_candidates
Step 7 → state.selected_page
Step 8 → state.context
Step 9-10 → state.answer
Step 11 → Final output
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class QueryIntent(str, Enum):
    """Query intent classification"""
    LOOKUP = "lookup"           # Find specific fact
    EXPLAIN = "explain"         # Explain concept
    SUMMARIZE = "summarize"     # Summarize section
    COMPARE = "compare"         # Compare concepts


@dataclass
class RetrievedChunkState:
    """Chunk state during retrieval"""
    chunk_id: str
    page: int
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageCandidateState:
    """Page candidate during selection"""
    page: int
    score: float
    chunks: List[RetrievedChunkState] = field(default_factory=list)
    semantic_score: float = 0.0
    structural_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "structural_score": self.structural_score,
            "chunk_count": len(self.chunks),
        }


@dataclass
class SelectedPageState:
    """Selected page with full context"""
    page: int
    chapter: Optional[str]
    section: Optional[str]
    subsection: Optional[str]
    title: Optional[str]
    chunks: List[RetrievedChunkState] = field(default_factory=list)
    score: float = 0.0
    
    def to_citation(self) -> Dict[str, Any]:
        """Convert to citation format"""
        citation = {
            "page": self.page,
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "title": self.title,
        }
        # Remove None values
        return {k: v for k, v in citation.items() if v is not None}


class AgentState:
    """
    Complete agent state for a single query execution.
    """
    
    def __init__(self):
        # Input
        self.query: str = ""
        
        # Step 2
        self.normalized_query: str = ""
        
        # Step 3
        self.intent: QueryIntent = QueryIntent.LOOKUP
        
        # Step 4-5
        self.retrieved_chunks: List[RetrievedChunkState] = []
        
        # Step 6
        self.page_candidates: List[PageCandidateState] = []
        
        # Step 7
        self.selected_page: Optional[SelectedPageState] = None
        
        # Step 8
        self.context: str = ""
        
        # Step 9-10
        self.answer: str = ""
        self.answer_valid: bool = False
        self.validation_attempts: int = 0
        
        # Metadata
        self.project: Optional[str] = None
        self.execution_time_ms: float = 0.0
        self.error: Optional[str] = None
    
    def get_intent_config(self) -> Dict[str, Any]:
        """
        Get configuration based on intent.
        
        Intent affects:
        - topK for retrieval
        - max_pages to consider
        - context length
        """
        configs = {
            QueryIntent.LOOKUP: {
                "top_k": 30,
                "max_pages": 3,
                "max_context_length": 4000,
            },
            QueryIntent.EXPLAIN: {
                "top_k": 50,
                "max_pages": 5,
                "max_context_length": 8000,
            },
            QueryIntent.SUMMARIZE: {
                "top_k": 100,
                "max_pages": 10,
                "max_context_length": 12000,
            },
            QueryIntent.COMPARE: {
                "top_k": 80,
                "max_pages": 8,
                "max_context_length": 10000,
            },
        }
        return configs.get(self.intent, configs[QueryIntent.LOOKUP])
    
    def has_selected_page(self) -> bool:
        """Check if agent successfully selected a page"""
        return self.selected_page is not None
    
    def is_valid_to_answer(self) -> bool:
        """
        RULE 1: Agent can only answer if page is selected
        """
        return self.has_selected_page()
    
    def get_page_candidates_summary(self) -> str:
        """Get summary of page candidates for debugging"""
        if not self.page_candidates:
            return "No page candidates"
        
        lines = []
        for cand in self.page_candidates[:5]:  # Top 5
            lines.append(f"  Page {cand.page}: score={cand.score:.3f} chunks={len(cand.chunks)}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dict for logging/debugging"""
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "intent": self.intent.value,
            "retrieved_chunks_count": len(self.retrieved_chunks),
            "page_candidates_count": len(self.page_candidates),
            "selected_page": self.selected_page.to_citation() if self.selected_page else None,
            "context_length": len(self.context),
            "answer_length": len(self.answer),
            "answer_valid": self.answer_valid,
            "validation_attempts": self.validation_attempts,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }
    
    def __repr__(self) -> str:
        return f"""AgentState(
  query='{self.query[:50]}...'
  intent={self.intent.value}
  chunks={len(self.retrieved_chunks)}
  candidates={len(self.page_candidates)}
  selected_page={self.selected_page.page if self.selected_page else None}
  answer_valid={self.answer_valid}
)"""
