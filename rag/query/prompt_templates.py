"""
LangChain PromptTemplates for RAG Pipeline
===========================================

Clean, reusable prompt templates using LangChain.
No hardcoded strings - all templates are parametrized.

Used in:
- page_response.py - build RAG context prompts
- agent/answer_generator.py - generate final answers
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from typing import Dict, Any, Optional


# ===== RAG CONTEXT BUILDING TEMPLATE =====

RAG_CONTEXT_TEMPLATE = """You are a helpful assistant that answers questions based on provided document context.

CONTEXT:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "I don't have this information"
- Cite the specific section/page when possible
- Be concise and accurate"""

rag_context_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_CONTEXT_TEMPLATE,
)


# ===== DOCUMENT SUMMARIZATION TEMPLATE =====

SUMMARIZATION_TEMPLATE = """Summarize the following document content in 2-3 sentences.
Focus on main topic, key points, and purpose.

Document content:
{content}

Summary:"""

summarization_prompt = PromptTemplate(
    input_variables=["content"],
    template=SUMMARIZATION_TEMPLATE,
)


# ===== PAGE STRUCTURE ANALYSIS TEMPLATE =====

STRUCTURE_ANALYSIS_TEMPLATE = """Analyze the following page content and extract:
1. Chapter number (if any)
2. Section number and title
3. Main heading/title of this section
4. Key topics covered

Page content:
{page_content}

Return as JSON with keys: chapter, section, title, topics"""

structure_analysis_prompt = PromptTemplate(
    input_variables=["page_content"],
    template=STRUCTURE_ANALYSIS_TEMPLATE,
)


# ===== INTENT CLASSIFICATION TEMPLATE =====

INTENT_TEMPLATE = """Classify the query intent into one of these categories:
- FACTUAL: Asking for specific facts or definitions
- ANALYTICAL: Asking for analysis, comparison, or explanation
- PROCEDURAL: Asking how to do something
- SUMMARY: Asking to summarize content

Query: {query}

Intent:"""

intent_prompt = PromptTemplate(
    input_variables=["query"],
    template=INTENT_TEMPLATE,
)


# ===== ANSWER GENERATION TEMPLATE =====

ANSWER_GENERATION_TEMPLATE = """Based on the retrieved pages, answer the following question comprehensively.

Question: {question}

Retrieved context from pages:
{retrieved_pages}

Requirements:
- Provide a complete answer
- Include relevant details from context
- Format response for clarity (use sections if needed)
- Always cite source pages

Answer:"""

answer_generation_prompt = PromptTemplate(
    input_variables=["question", "retrieved_pages"],
    template=ANSWER_GENERATION_TEMPLATE,
)


# ===== PAGE-AWARE RESPONSE TEMPLATE =====

PAGE_AWARE_TEMPLATE = """You are answering based on document pages.

Question: {question}

Retrieved pages:
{pages_context}

Instructions:
1. Answer the question comprehensively
2. Reference specific pages: "Page X shows..."
3. Group related information logically
4. If answer spans multiple pages, indicate progression

Answer:"""

page_aware_prompt = PromptTemplate(
    input_variables=["question", "pages_context"],
    template=PAGE_AWARE_TEMPLATE,
)

def get_rag_context_prompt() -> PromptTemplate:
    """Get RAG context prompt template"""
    return rag_context_prompt


def get_summarization_prompt() -> PromptTemplate:
    """Get summarization prompt template"""
    return summarization_prompt


def get_structure_analysis_prompt() -> PromptTemplate:
    """Get structure analysis prompt template"""
    return structure_analysis_prompt


def get_intent_prompt() -> PromptTemplate:
    """Get intent classification prompt template"""
    return intent_prompt


def get_answer_generation_prompt() -> PromptTemplate:
    """Get answer generation prompt template"""
    return answer_generation_prompt


def get_page_aware_prompt() -> PromptTemplate:
    """Get page-aware response prompt template"""
    return page_aware_prompt


def format_rag_context(context: str, question: str) -> str:
    """
    Format RAG context prompt with variables.
    
    Args:
        context: Document context
        question: User question
        
    Returns:
        Formatted prompt string
    """
    return rag_context_prompt.format(context=context, question=question)


def format_answer_generation(question: str, retrieved_pages: str) -> str:
    """
    Format answer generation prompt with variables.
    
    Args:
        question: User question
        retrieved_pages: Retrieved pages formatted for context
        
    Returns:
        Formatted prompt string
    """
    return answer_generation_prompt.format(
        question=question,
        retrieved_pages=retrieved_pages
    )
