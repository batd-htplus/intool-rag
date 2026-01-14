from typing import List
from rag.config import config
from rag.logging import logger

def format_context_with_metadata(results: List, max_results: int = 5) -> str:
    """Format retrieved results with metadata and relevance scores
    
    Args:
        results: List of query results
        max_results: Maximum number of results to include (default: 5 for shorter context)
    """
    if not results:
        return ""
    
    limited_results = results[:max_results]
    
    context_parts = []
    for i, result in enumerate(limited_results, 1):
        score = getattr(result, 'score', 0)
        metadata = getattr(result, 'metadata', {})
        text = getattr(result, 'text', '')
        
        if len(text) > 1500:
            text = text[:1500] + "..."
        
        source_info = f"[Source {i}, Relevance: {score:.2%}]"
        if metadata:
            doc_name = metadata.get("filename") or metadata.get("doc_name", "Unknown")
            source_info += f" ({doc_name})"
        
        context_parts.append(f"{source_info}\n{text}")
    
    return "\n\n---\n\n".join(context_parts)

def build_prompt(context: str, question: str) -> str:
    """Build RAG prompt from context and question"""
    
    prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Instructions:
- Read the context carefully
- Extract relevant information to answer the question
- If the context contains the answer, provide a clear and complete answer
- If the context does NOT contain enough information, respond with: "I don't have this information in the provided documents."
- Answer in the same language as the question

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    return prompt

def build_chat_prompt(context: str, chat_history: List[dict], question: str) -> str:
    """Build prompt with chat history"""
    
    # Format chat history (last 5 messages for context)
    history_text = ""
    for msg in chat_history[-5:]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"
    
    prompt = f"""You are a helpful AI assistant. Answer the following question based on the provided context and conversation history.

If the context does not contain information to answer the question, respond with: "I don't have this information in the provided documents."

Do NOT make up information or use external knowledge. Only use the context and conversation history below.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION: {question}

YOUR RESPONSE:"""
    
    return prompt
