from typing import List, Optional
import re
import hashlib
import unicodedata
from rag.config import config
from rag.logging import logger


# =====================================================
# 1. TEXT CLEANING (JA / VI / EN SAFE)
# =====================================================

INVISIBLE_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\uFEFF]'
)

ALLOWED_CHARS_RE = re.compile(
    r'[^\u0020-\u007E'
    r'\u00A0-\u024F'
    r'\u3040-\u309F'
    r'\u30A0-\u30FF'
    r'\u4E00-\u9FFF'
    r'\u3000-\u303F]'
)

REPEATED_PUNCT_RE = re.compile(r'([.!?,:;]){2,}')
BROKEN_LINE_RE = re.compile(r'(?<![.!?:;])\n(?!\n)')


def clean_text_multilang(text: str) -> str:
    """
    Clean noisy text (PDF / OCR safe)
    Preserve Japanese, Vietnamese, English
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE_RE.sub("", text)
    text = ALLOWED_CHARS_RE.sub(" ", text)
    text = BROKEN_LINE_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# 2. SENTENCE & KEYWORD UTILITIES
# =====================================================

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?;:])\s+|\n+')
WORD_RE = re.compile(r'\b\w+\b')


def _split_sentences(text: str) -> List[str]:
    text = clean_text_multilang(text)
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) >= 20]


def _extract_keywords(query: str) -> List[str]:
    words = WORD_RE.findall(query.lower())
    return [w for w in words if len(w) > 2]


# =====================================================
# 3. STRUCTURED DATA HANDLING (GENERIC)
# =====================================================

TABLE_PATTERN = re.compile(r'\[TABLE\].*?\[/TABLE\]', re.DOTALL)


def _extract_table_blocks(text: str) -> List[str]:
    """
    Extract all table blocks from text.
    
    Args:
        text: Text that may contain table blocks
        
    Returns:
        List of table block strings (including markers)
    """
    if not text or "[TABLE]" not in text:
        return []
    return TABLE_PATTERN.findall(text)


def _is_structured_data(metadata: dict) -> bool:
    """
    Check if chunk contains structured data (tables, lists, etc.).
    
    Args:
        metadata: Chunk metadata
        
    Returns:
        True if chunk contains structured data
    """
    if not metadata:
        return False
    return (
        metadata.get("has_table", False) or
        metadata.get("doc_type") == "table" or
        metadata.get("has_list", False) or
        metadata.get("doc_type") == "list"
    )


def _preserve_structured_content(
    text: str,
    max_chars: int,
    metadata: Optional[dict] = None
) -> str:
    """
    Preserve structured content (tables, lists) entirely.
    
    This ensures structured data is not fragmented by sentence-based extraction.
    
    Args:
        text: Source text
        max_chars: Maximum characters to preserve
        metadata: Optional metadata to check for structured data
        
    Returns:
        Preserved structured content or truncated text
    """
    if not text or not text.strip():
        return ""
    
    # Check for table blocks
    table_blocks = _extract_table_blocks(text)
    if table_blocks:
        total_length = sum(len(block) for block in table_blocks)
        
        if total_length <= max_chars:
            result_parts = []
            remaining = max_chars - total_length
            
            before_first = text[:text.find(table_blocks[0])].strip()
            if before_first and remaining > 100:
                context_before = before_first[:min(100, remaining // 2)]
                result_parts.append(context_before)
                remaining -= len(context_before)
            
            result_parts.extend(table_blocks)
            
            after_last = text[text.rfind(table_blocks[-1]) + len(table_blocks[-1]):].strip()
            if after_last and remaining > 50:
                context_after = after_last[:min(remaining, 50)]
                result_parts.append(context_after)
            
            return "\n".join(result_parts)
        else:
            result_parts = []
            remaining = max_chars
            for block in table_blocks:
                if len(block) <= remaining:
                    result_parts.append(block)
                    remaining -= len(block)
                else:
                    if not result_parts:
                        result_parts.append(block[:remaining])
                    break
            return "\n".join(result_parts) if result_parts else table_blocks[0][:max_chars]
    
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


# =====================================================
# 4. SENTENCE SCORING (PRECISION CORE)
# =====================================================

def _score_sentence(sentence: str, keywords: List[str]) -> float:
    """
    Score sentence by keyword relevance with improved weighting.
    
    Args:
        sentence: Sentence to score
        keywords: List of query keywords
        
    Returns:
        Relevance score (higher = more relevant)
    """
    if not sentence or not keywords:
        return 0.0
    
    s = sentence.lower()
    hits = sum(1 for w in keywords if w in s)
    if hits == 0:
        return 0.0

    word_count = max(len(sentence.split()), 1)
    density = hits / word_count
    
    score = (hits * 2.5) + (density * 1.5)
    
    return score


def _extract_evidence(
    text: str,
    query: str,
    max_chars: int,
    min_score: float = 1.0
) -> str:
    """
    Extract highest-signal sentences as evidence with context window preservation.
    
    Features:
    - Query-aware sentence scoring
    - Context window preservation (keeps surrounding sentences for coherence)
    - Structured data preservation (tables, lists)
    - Fallback to first sentences if no keywords match
    - Efficient character limit management
    
    Args:
        text: Source text to extract from
        query: Query text for relevance scoring
        max_chars: Maximum characters to extract
        min_score: Minimum relevance score threshold
        
    Returns:
        Extracted evidence text optimized for context
    """
    if not text or not text.strip():
        return ""
    
    # Preserve structured content first
    if "[TABLE]" in text and "[/TABLE]" in text:
        preserved = _preserve_structured_content(text, max_chars)
        if preserved:
            return preserved
    
    sentences = _split_sentences(text)
    if not sentences:
        cleaned = clean_text_multilang(text)
        return cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")

    keywords = _extract_keywords(query) if query else []
    if not keywords:
        selected = []
        total_len = 0
        for s in sentences:
            if total_len + len(s) > max_chars:
                break
            selected.append(s)
            total_len += len(s)
        return " ".join(selected) if selected else sentences[0][:max_chars]

    scored = [
        (_score_sentence(s, keywords), s, idx)
        for idx, s in enumerate(sentences)
    ]

    scored_filtered = [x for x in scored if x[0] >= min_score]
    
    if not scored_filtered:
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        if scored_sorted:
            selected = []
            total_len = 0
            for _, s, _ in scored_sorted[:3]:
                if total_len + len(s) > max_chars:
                    break
                selected.append(s)
                total_len += len(s)
            return " ".join(selected) if selected else sentences[0][:max_chars]
        selected = []
        total_len = 0
        for s in sentences[:3]:
            if total_len + len(s) > max_chars:
                break
            selected.append(s)
            total_len += len(s)
        return " ".join(selected) if selected else sentences[0][:max_chars]
    
    scored_filtered.sort(key=lambda x: x[0], reverse=True)

    selected_indices = set()
    selected_with_idx = []
    total_len = 0

    for score, sentence, idx in scored_filtered:
        if idx in selected_indices:
            continue
        if total_len + len(sentence) > max_chars:
            break
        
        selected_with_idx.append((idx, sentence))
        selected_indices.add(idx)
        total_len += len(sentence)
        
        if idx > 0 and idx - 1 not in selected_indices:
            prev_sentence = sentences[idx - 1]
            if total_len + len(prev_sentence) <= max_chars:
                selected_with_idx.insert(-1, (idx - 1, prev_sentence))
                selected_indices.add(idx - 1)
                total_len += len(prev_sentence)

    if selected_with_idx:
        selected_with_idx.sort(key=lambda x: x[0])
        final_selected = [s for _, s in selected_with_idx]
        
        # Ensure we don't exceed max_chars
        result = " ".join(final_selected)
        if len(result) > max_chars:
            result = result[:max_chars]
            last_space = result.rfind(" ")
            if last_space > max_chars * 0.8:  
                result = result[:last_space] + "..."
            else:
                result = result + "..."
        return result

    return sentences[0][:max_chars] if sentences else ""


# =====================================================
# 5. DEDUPLICATION
# =====================================================

def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _deduplicate_results(results: List) -> List:
    """
    Deduplicate by document + page + content hash.
    
    This prevents duplicate content from:
    - Same document across different retrievals
    - Footer/header repetition across pages
    - Identical chunks from different sources
    
    Args:
        results: List of retrieval results to deduplicate
        
    Returns:
        Deduplicated list of results
    """
    seen = set()
    unique = []

    for r in results:
        metadata = getattr(r, "metadata", {}) or {}
        doc_id = (
            metadata.get("doc_id")
            or metadata.get("filename")
            or metadata.get("doc_name")
            or ""
        )

        text = getattr(r, "text", "")
        if not text:
            continue

        page = metadata.get("page")
        key = (doc_id, page, _stable_hash(text[:300]))
        if key in seen:
            continue

        seen.add(key)
        unique.append(r)

    return unique


# =====================================================
# 6. CONTEXT FORMATTING
# =====================================================

def _extract_snippet(
    text: str,
    metadata: dict,
    query: str,
    max_text_length: int
) -> str:
    """
    Extract snippet from text based on content type.
    
    For structured data (tables, lists), preserve entire structure.
    For regular text, use evidence-based extraction.
    
    Args:
        text: Source text
        metadata: Chunk metadata
        query: Query text for relevance scoring
        max_text_length: Maximum snippet length
        
    Returns:
        Extracted snippet
    """
    if not text or not text.strip():
        return ""
    
    # Preserve structured content entirely
    if _is_structured_data(metadata):
        preserved = _preserve_structured_content(text, max_text_length, metadata)
        if preserved:
            return preserved
    
    # Regular text: use evidence-based extraction
    snippet = _extract_evidence(
        text=text,
        query=query,
        max_chars=max_text_length,
        min_score=1.0
    )

    if not snippet or not snippet.strip():
        snippet = text[:max_text_length] + ("..." if len(text) > max_text_length else "")
    
    return snippet


def _format_document_header(
    idx: int,
    metadata: dict,
    default_name: str = "Document"
) -> str:
    """
    Format document header with metadata.
    
    Args:
        idx: Document index
        metadata: Document metadata
        default_name: Default document name
        
    Returns:
        Formatted header string
    """
    doc_name = (
        metadata.get("source") or
        metadata.get("filename") or
        metadata.get("doc_name") or
        f"{default_name} {idx}"
    )
    
    parts = [f"[{idx}] {doc_name}"]
    
    if metadata.get("page"):
        parts.append(f"(Page {metadata.get('page')})")
    
    if _is_structured_data(metadata):
        parts.append("[STRUCTURED DATA]")
    
    return " ".join(parts).strip()


def format_context_with_metadata(
    results: List,
    max_results: int = None,
    max_text_length: int = None,
    query: str = None
) -> str:
    """
    Build high-precision RAG context with optimized formatting.
    
    Features:
    - Smart evidence extraction based on query relevance
    - Structured document references with metadata
    - Preserves structured data (tables, lists) entirely
    - Efficient token usage
    - Deduplication to avoid redundant information
    - Results sorted by relevance score (highest first)
    
    Args:
        results: List of retrieval results
        max_results: Maximum number of results to include
        max_text_length: Maximum text length per result
        query: Optional query text for relevance scoring
        
    Returns:
        Formatted context string
    """
    if not results:
        return ""
    
    query = query or ""
    max_results = max_results or config.CONTEXT_MAX_RESULTS
    max_text_length = max_text_length or config.CONTEXT_MAX_TEXT_LENGTH

    results = _deduplicate_results(results)
    
    if not results:
        return ""
    
    # Sort by relevance score (highest first)
    if hasattr(results[0], "score"):
        results = sorted(results, key=lambda x: getattr(x, "score", 0.0), reverse=True)
    elif isinstance(results[0], dict) and "score" in results[0]:
        results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    
    results = results[:max_results]

    blocks = []
    for idx, r in enumerate(results, 1):
        # Extract text
        if hasattr(r, "text"):
            text = r.text
        elif isinstance(r, dict):
            text = r.get("text", "")
        else:
            text = str(r) if r else ""
        
        if not text or not text.strip():
            continue

        # Extract metadata
        if hasattr(r, "metadata"):
            metadata = r.metadata or {}
        elif isinstance(r, dict):
            metadata = r.get("metadata", {}) or {}
        else:
            metadata = {}

        # Extract snippet based on content type
        snippet = _extract_snippet(text, metadata, query, max_text_length)
        
        if not snippet or not snippet.strip():
            continue

        # Format header
        header = _format_document_header(idx, metadata)
        
        blocks.append(f"{header}\n{snippet}")

    return "\n\n".join(blocks)


# =====================================================
# 7. PROMPT BUILDING (GENERIC, CONTEXT-AWARE)
# =====================================================

def _has_structured_data(context: str) -> bool:
    """
    Check if context contains structured data markers.
    
    Args:
        context: Context string
        
    Returns:
        True if structured data is present
    """
    if not context:
        return False
    return (
        "[TABLE]" in context or
        "[/TABLE]" in context or
        "[STRUCTURED DATA]" in context
    )


def _build_base_instructions() -> List[str]:
    """
    Build base instructions for LLM (generic, not domain-specific).
    
    Returns:
        List of instruction strings
    """
    return [
        "You are an expert assistant that answers questions based ONLY on the provided context.",
        "",
        "## ANTI-HALLUCINATION CONTRACT:",
        "1. You MUST use ONLY the exact text from the context below",
        "2. You MUST NOT infer, guess, reconstruct, or add ANY information not explicitly present in the context",
        "3. You MUST NOT use your general knowledge - only use what is written in the context",
        "4. If the answer is not in the context, respond exactly: \"I don't have this information in the provided context.\"",
        "",
        "## Instructions:",
        "1. Extract information directly from the context - do not paraphrase or interpret",
        "2. For structured data (tables, lists): extract exact values, do not calculate or reconstruct",
        "3. Cite the exact source when possible (e.g., 'According to the table...', 'As shown in the document...')",
        "4. Be precise and factual - copy values exactly as they appear",
    ]


def _build_structured_data_instructions() -> List[str]:
    """
    Build instructions for handling structured data (tables, lists, etc.).
    
    Returns:
        List of instruction strings for structured data
    """
    return [
        "",
        "## Structured Data Analysis Rules:",
        "1. The context contains structured data (tables, lists, etc.) - you MUST extract data directly from structured elements",
        "2. Structured data has HIGHEST PRIORITY - it contains factual, structured information",
        "3. Read ALL rows/items in structured data to find complete information",
        "4. Extract EXACT values from structured data - do NOT calculate, infer, or reconstruct",
        "5. Structured data is the authoritative source for factual information",
        "6. DO NOT say 'I don't have this information' if the answer exists in structured data",
        "7. When showing structured data, preserve the original format (use | separators for tables)",
        "8. DO NOT add explanations or interpretations - only extract and cite",
        "",
        "## Answer Format:",
        "- Extract → Cite → Answer",
        "- Example: 'Yes. According to the table: Item A: $100, Item B: $200'",
        "- DO NOT: 'Yes, there are items' (too vague, no exact values)",
    ]


def build_prompt(context: str, question: str) -> str:
    """
    Build optimized factual prompt (hallucination-safe, multilingual-aware, performance-optimized).
    
    Features:
    - Clear instructions to prevent hallucination
    - Structured format for better LLM understanding
    - Support for multilingual responses
    - Efficient token usage
    - Context-aware formatting
    - Generic structured data handling (not domain-specific)
    
    Args:
        context: Retrieved context from documents
        question: User question
        
    Returns:
        Optimized prompt string ready for LLM
    """
    if not context or not context.strip():
        return f"""You are a helpful assistant. Answer the following question based on your knowledge.

## Question:
{question}

## Answer:"""
    
    has_structured = _has_structured_data(context)
    
    prompt_parts = _build_base_instructions()
    
    if has_structured:
        prompt_parts.extend(_build_structured_data_instructions())
        prompt_parts.append("5. You can respond in the same language as the question")
    else:
        prompt_parts.append("5. You can respond in the same language as the question")
    
    prompt_parts.extend([
        "",
        "## Context:",
        context,
        "",
        "## Question:",
        question,
        "",
        "## Answer:"
    ])
    
    return "\n".join(prompt_parts)


def build_chat_prompt(
    context: str,
    chat_history: List[dict],
    question: str,
    max_history: int = None
) -> str:
    """
    Build optimized chat prompt with bounded history.
    
    Features:
    - Maintains conversation context efficiently
    - Bounded history to prevent token overflow
    - Clear instructions for context usage
    - Support for multilingual conversations
    - Smart history truncation
    - Generic structured data handling
    
    Args:
        context: Retrieved context from documents
        chat_history: List of conversation messages
        question: Current user question
        max_history: Maximum number of history messages to include
        
    Returns:
        Optimized chat prompt string ready for LLM
    """
    max_history = max_history or config.CHAT_HISTORY_MAX_MESSAGES

    history_lines = []
    for msg in chat_history[-max_history:]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        
        if len(content) > 300:
            truncated = content[:300]
            last_period = truncated.rfind(".")
            last_question = truncated.rfind("?")
            last_exclamation = truncated.rfind("!")
            last_boundary = max(last_period, last_question, last_exclamation)
            if last_boundary > 200:
                content = content[:last_boundary + 1] + "..."
            else:
                content = content[:300] + "..."
        
        if role == "user":
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")
        else:
            history_lines.append(f"{role.capitalize()}: {content}")

    history_text = "\n".join(history_lines) if history_lines else "No previous conversation."

    if not context or not context.strip():
        return f"""You are a helpful assistant in a conversation. Answer the following question based on your knowledge and conversation history.

## Conversation History:
{history_text}

## Current Question:
{question}

## Answer:"""

    has_structured = _has_structured_data(context)
    
    prompt_parts = [
        "You are an expert assistant in a conversation. Answer questions based ONLY on the provided context and conversation history.",
        "",
    ]
    prompt_parts.extend(_build_base_instructions())
    prompt_parts.append("5. Maintain conversation flow and coherence")
    
    if has_structured:
        prompt_parts.extend(_build_structured_data_instructions())
        prompt_parts.append("6. You can respond in the same language as the question")
    else:
        prompt_parts.append("6. You can respond in the same language as the question")
    
    prompt_parts.extend([
        "",
        "## Context:",
        context,
        "",
        "## Conversation History:",
        history_text,
        "",
        "## Current Question:",
        question,
        "",
        "## Answer:"
    ])
    
    return "\n".join(prompt_parts)
