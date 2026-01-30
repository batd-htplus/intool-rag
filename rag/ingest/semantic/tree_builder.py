"""
Semantic Tree Builder
=====================

Responsibility:
- Build a semantic tree for a document using LLM-based analysis
- NO heuristic / regex logic
- Delegates ALL semantic understanding to llm.semantic

Flow:
1. Combine document pages
2. Ask LLM to analyze semantic structure
3. Convert LLM output → SemanticNode objects
4. Persist PageIndex
"""

from datetime import datetime
from typing import List, Dict, Any

from rag.logging import logger
from rag.ingest.schemas import (
    SemanticNode,
    PageIndex,
    NodeLevel,
    save_page_index,
)
from rag.llm.semantic import analyze_document
from rag.config import config

class SemanticTreeBuilder:
    """
    Build semantic tree using LLM-powered structure analysis.
    """

    def __init__(self) -> None:
        self._next_node_id = 0

    def _generate_node_id(self) -> str:
        node_id = f"{self._next_node_id:04d}"
        self._next_node_id += 1
        return node_id

    async def build(
        self,
        doc_id: str,
        source_filename: str,
        pages_data: List[Dict[str, Any]],
        language: str = "en",
    ) -> PageIndex:
        """
        Build semantic tree for a document.

        Args:
            doc_id: unique document id
            source_filename: original filename
            pages_data: list of page dicts {page, clean_text}
            language: document language

        Returns:
            PageIndex
        """
        logger.info(f"[INGEST][SEMANTIC] Building semantic tree for {doc_id}")

        full_text = self._prepare_full_text(pages_data)

        response = await analyze_document(full_text)
        logger.info(f"[INGEST][cleaned_response] {doc_id}: {response}")

        sections = response.get("sections", [])

        if not sections:
            logger.warning("[INGEST][SEMANTIC] No sections returned by LLM")

        nodes: List[SemanticNode] = []
        parent_stack: List[tuple[str, str]] = []

        for section in sections:
            title = section.get("title", "").strip()
            summary = section.get("summary", "").strip()
            level = section.get("level", "paragraph").lower()
            page_index = section.get("page_index", 1)

            if not title or not summary:
                continue

            node_id = self._generate_node_id()
            parent_id = self._resolve_parent_id(level, parent_stack)

            node = SemanticNode(
                node_id=node_id,
                title=title,
                summary=summary,
                level=NodeLevel(level),
                page_index=page_index,
                parent_id=parent_id,
                children=[],
            )

            if parent_id:
                parent = next(n for n in nodes if n.node_id == parent_id)
                parent.children.append(node_id)

            nodes.append(node)
            parent_stack.append((node_id, level))

        page_index = PageIndex(
            doc_id=doc_id,
            source_filename=source_filename,
            created_at=datetime.utcnow().isoformat(),
            nodes=nodes,
            root_node_id=nodes[0].node_id if nodes else "0000",
            page_count=len(pages_data),
            node_count=len(nodes),
            language=language,
        )

        logger.info(
            f"[INGEST][SEMANTIC] ✓ Built semantic tree "
            f"({len(nodes)} nodes)"
        )

        return page_index

    def _prepare_full_text(self, pages_data: List[Dict[str, Any]]) -> str:
        text = []
        for page in pages_data:
            page_num = page.get("page", 1)
            content = page.get("clean_text", "")
            text.append(f"\n[PAGE {page_num}]\n{content}")
        return "\n".join(text)

    def _resolve_parent_id(
        self,
        level: str,
        parent_stack: List[tuple[str, str]],
    ) -> str | None:
        hierarchy = ["chapter", "section", "subsection", "paragraph"]

        if level not in hierarchy:
            return None

        parent_level = hierarchy[hierarchy.index(level) - 1] if level != "chapter" else None

        while parent_stack:
            node_id, node_level = parent_stack[-1]
            if node_level == parent_level:
                return node_id
            parent_stack.pop()

        return None
