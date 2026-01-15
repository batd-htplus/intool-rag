from typing import List
import re
import hashlib
import unicodedata
from rag.config import config
from rag.logging import logger


# =====================================================
# 1. TEXT CLEANING (JA / VI / EN SAFE)
# =====================================================

# Invisible / zero-width characters
INVISIBLE_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\uFEFF]'
)

# Allow:
# - Basic Latin
# - Latin Extended (Vietnamese)
# - Hiragana, Katakana
# - Kanji
# - CJK punctuation
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

    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible chars
    text = INVISIBLE_RE.sub("", text)

    # Remove non-allowed chars
    text = ALLOWED_CHARS_RE.sub(" ", text)

    # Fix broken OCR line breaks
    text = BROKEN_LINE_RE.sub(" ", text)

    # Normalize punctuation
    text = REPEATED_PUNCT_RE.sub(r"\1", text)

    # Normalize whitespace
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
# 3. SENTENCE SCORING (PRECISION CORE)
# =====================================================

def _score_sentence(sentence: str, keywords: List[str]) -> float:
    """
    Score by keyword hit count + density
    """
    s = sentence.lower()
    hits = sum(1 for w in keywords if w in s)
    if hits == 0:
        return 0.0

    density = hits / max(len(sentence.split()), 1)
    return hits * 2.0 + density


def _extract_evidence(
    text: str,
    query: str,
    max_chars: int,
    min_score: float = 1.0
) -> str:
    """
    Extract highest-signal sentences as evidence
    Falls back to first sentences if no keywords match
    """
    if not text or not text.strip():
        return ""
    
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
        (_score_sentence(s, keywords), s)
        for s in sentences
    ]

    # Filter noise
    scored_filtered = [x for x in scored if x[0] >= min_score]
    if not scored_filtered:
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        if scored_sorted:
            selected = []
            total_len = 0
            for _, s in scored_sorted[:3]:
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
    
    scored = scored_filtered

    # Sort by relevance
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    total_len = 0

    for _, sentence in scored:
        if sentence in selected:
            continue
        if total_len + len(sentence) > max_chars:
            break
        selected.append(sentence)
        total_len += len(sentence)

    return " ".join(selected) if selected else sentences[0][:max_chars]


# =====================================================
# 4. DEDUPLICATION (STABLE)
# =====================================================

def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _deduplicate_results(results: List) -> List:
    """
    Deduplicate by document + content hash
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

        key = (doc_id, _stable_hash(text[:300]))
        if key in seen:
            continue

        seen.add(key)
        unique.append(r)

    return unique


# =====================================================
# 5. CONTEXT BUILDER (NO NOISE)
# =====================================================

def format_context_with_metadata(
    results: List,
    max_results: int = None,
    max_text_length: int = None,
    query: str = None
) -> str:
    """
    Build high-precision RAG context
    """
    if not results:
        return ""
    
    # Query is optional - if not provided, use first sentences
    query = query or ""

    max_results = max_results or config.CONTEXT_MAX_RESULTS
    max_text_length = max_text_length or config.CONTEXT_MAX_TEXT_LENGTH

    results = _deduplicate_results(results)[:max_results]

    blocks = []
    for idx, r in enumerate(results, 1):
        if hasattr(r, "text"):
            text = r.text
        elif isinstance(r, dict):
            text = r.get("text", "")
        else:
            text = str(r) if r else ""
        
        if not text or not text.strip():
            continue

        if hasattr(r, "metadata"):
            metadata = r.metadata or {}
        elif isinstance(r, dict):
            metadata = r.get("metadata", {}) or {}
        else:
            metadata = {}

        snippet = _extract_evidence(
            text=text,
            query=query,
            max_chars=max_text_length,
            min_score=1.0
        )

        if not snippet or not snippet.strip():
            cleaned = clean_text_multilang(text)
            snippet = cleaned[:max_text_length] + ("..." if len(cleaned) > max_text_length else "")

        doc_name = metadata.get("source") or metadata.get("filename") or metadata.get("doc_name") or f"Document {idx}"
        header = f"[{idx}] {doc_name}".strip()

        blocks.append(f"{header}\n{snippet}")

    return "\n\n".join(blocks)


# =====================================================
# 6. PROMPT BUILDERS (STRICT & SAFE)
# =====================================================

def build_prompt(context: str, question: str) -> str:
    """
    Strict factual prompt (hallucination-safe)
    """
    return f"""You are a precise question-answering assistant.

Use ONLY the context below.
Do NOT infer or guess.
If the answer is not explicitly stated, reply exactly:
"I don't have this information."

Context:
{context}

Question: {question}
Answer:"""


def build_chat_prompt(
    context: str,
    chat_history: List[dict],
    question: str,
    max_history: int = None
) -> str:
    """
    Chat prompt with bounded history (noise-safe)
    """
    max_history = max_history or config.CHAT_HISTORY_MAX_MESSAGES

    history_lines = []
    for msg in chat_history[-max_history:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        history_lines.append(f"{role}: {content}")

    history_text = "\n".join(history_lines)

    return f"""You are a precise question-answering assistant.

Use ONLY the context and history below.
Do NOT infer or guess.
If the answer is not explicitly stated, reply exactly:
"I don't have this information."

Context:
{context}

History:
{history_text}

Question: {question}
Answer:"""
