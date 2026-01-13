from typing import List
from rag.config import config
from rag.logging import logger

def format_context_with_metadata(results: List) -> str:
    """Format retrieved results with metadata and relevance scores"""
    if not results:
        return ""
    
    context_parts = []
    for i, result in enumerate(results, 1):
        score = getattr(result, 'score', 0)
        metadata = getattr(result, 'metadata', {})
        text = getattr(result, 'text', '')
        
        # Format each source with score and metadata
        source_info = f"[Source {i}, Relevance: {score:.2%}]"
        if metadata:
            doc_name = metadata.get("filename") or metadata.get("doc_name", "Unknown")
            source_info += f" ({doc_name})"
        
        context_parts.append(f"{source_info}\n{text}")
    
    return "\n\n---\n\n".join(context_parts)

def build_prompt(context: str, question: str) -> str:
    """Build RAG prompt from context and question"""
    
    prompt = f"""You are a helpful AI assistant. Answer the following question ONLY based on the provided context.

If the context does not contain information to answer the question, respond with: "I don't have this information in the provided documents."

Do NOT make up information or use external knowledge. Only use the context below.

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
