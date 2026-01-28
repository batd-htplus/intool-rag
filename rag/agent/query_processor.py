"""
Query Processor
===========================

Normalize query (remove fillers, standardize)
Classify intent (lookup|explain|summarize|compare)
"""

from typing import Optional
import re
from rag.agent.state import AgentState, QueryIntent
from rag.logging import logger


class QueryNormalizer:
    """Normalize query by removing fillers and standardizing"""
    
    FILLER_WORDS = {
        "please", "could", "would", "can", "may",
        "tell", "me", "about", "what", "is", "the",
        "how", "explain", "describe", "discuss"
    }
    
    def remove_filler(self, query: str) -> str:
        """Remove filler words"""
        words = query.lower().split()
        cleaned = [w for w in words if w not in self.FILLER_WORDS]
        return " ".join(cleaned)
    
    async def normalize(self, state: AgentState) -> None:
        """Normalize query and update state"""
        normalized = self.remove_filler(state.query).strip()
        state.normalized_query = normalized if normalized else state.query
        logger.info(f"  → {state.normalized_query}")


class IntentClassifier:
    """Classify query intent using pattern matching"""
    
    PATTERNS = {
        QueryIntent.LOOKUP: [
            r'what|who|where|when|find|tell.*about|show|get',
        ],
        QueryIntent.EXPLAIN: [
            r'explain|how.*work|why|how to|describe|clarify|understand',
        ],
        QueryIntent.SUMMARIZE: [
            r'summarize|summary|overview|brief|recap|sum up|main',
        ],
        QueryIntent.COMPARE: [
            r'compare|difference|versus?|contrast|similar',
        ],
    }
    
    def classify(self, query: str) -> QueryIntent:
        """Classify query intent"""
        query_lower = query.lower()
        
        for intent, patterns in self.PATTERNS.items():
            pattern_str = "|".join(patterns)
            if re.search(pattern_str, query_lower):
                return intent
        
        return QueryIntent.LOOKUP
    
    async def classify_intent(self, state: AgentState) -> None:
        """Classify intent and update state"""
        
        state.intent = self.classify(state.normalized_query)
        config = state.get_intent_config()
        
        logger.info(
            f"  → {state.intent.value} "
            f"(top_k={config['top_k']}, max_pages={config['max_pages']})"
        )
