"""
Semantic Tree Builder (PageIndex Generator) - LLM-Powered
==========================================================

Purpose:
- Send document text to Gemini LLM for COMPLETE analysis
- LLM analyzes structure AND generates summaries
- Build hierarchical semantic tree from LLM output
- Produce page_index.json with complete structure

Pipeline (ALL powered by LLM):
1. Prepare document text from pages
2. Send to Gemini for structure analysis
3. Gemini returns: sections, hierarchy, content
4. For each section, ask Gemini to generate summary
5. Build semantic tree from LLM output
6. Save as page_index.json

CRITICAL PRINCIPLE:
- Everything is LLM-driven, NOT regex/heuristic-based
- Structure analysis, hierarchy detection, summaries ALL from LLM
- This ensures document semantics are properly captured
- No hardcoded rules - LLM understands context
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from rag.logging import logger
from rag.ingest.schemas import (
    SemanticNode, 
    PageIndex, 
    NodeLevel,
    save_page_index,
    save_chunks_index,
)
from rag.ingest.gemini.semantic_summarizer import GeminiSemanticSummarizer
from rag.ingest.gemini.prompts import (
    DOCUMENT_STRUCTURE_ANALYSIS_PROMPT,
)
from rag.llm.embedding_service import get_embedding_provider

class SemanticTreeBuilder:
    """
    Build complete semantic tree using LLM for ALL analysis.
    
    The LLM does ALL the work:
    - Analyze document structure (chapters, sections, subsections)
    - Identify hierarchy and relationships
    - Extract content for each section
    - Generate semantic summaries
    
    NOT regex/heuristic-based. LLM understands document semantics.
    """
    
    def __init__(self):
        self.summarizer = GeminiSemanticSummarizer()
        self.next_node_id = 0
    
    def _generate_node_id(self) -> str:
        """Generate next node ID (4-digit padded)"""
        node_id = f"{self.next_node_id:04d}"
        self.next_node_id += 1
        return node_id
    
    async def build_tree(
        self,
        doc_id: str,
        source_filename: str,
        pages_data: List[Dict[str, Any]],
        language: str = "en",
    ) -> PageIndex:
        """
        Build complete semantic tree by asking LLM to analyze document.
        
        Process:
        1. Combine pages into document text
        2. Send to Gemini: "Analyze this document structure"
        3. Gemini returns sections with hierarchy
        4. For each section: ask Gemini to generate summary
        5. Build tree from LLM output
        
        Args:
            doc_id: Document ID
            source_filename: Original filename
            pages_data: List of page data with text
            language: Document language
            
        Returns:
            PageIndex with LLM-generated structure
        """
        logger.info(f"[BUILD] Semantic tree generation (LLM-powered) for {doc_id}")
        
        full_text = self._prepare_full_text(pages_data)
        sections = await self._analyze_structure_with_llm(full_text)
        logger.info(f"[BUILD] Ssections {sections}")
        
        save_page_index(page_index, doc_id)

        nodes = []
        parent_stack = []
        
        for i, section in enumerate(sections):
            logger.info(f"[BUILD] Processing section {section}")

            level = section.get("level", "paragraph")
            title = section.get("title", "")
            page = section.get("page_index", 1)
            summary =  section.get("summary", "")
            
            if not title or not summary.strip():
                continue
            
            parent_prefix = ""
            if parent_stack:
                parent_level = self._get_parent_level(level)
                while parent_stack and parent_stack[-1][1] != parent_level:
                    parent_stack.pop()
                
                if parent_stack:
                    parent_node = next(
                        (n for n in nodes if n.node_id == parent_stack[-1][0]), None
                    )
            
            node_id = self._generate_node_id()
            parent_id = None
            
            if parent_stack:
                parent_level = self._get_parent_level(level)
                while parent_stack and parent_stack[-1][1] != parent_level:
                    parent_stack.pop()
                
                if parent_stack:
                    parent_id = parent_stack[-1][0]
            
            node = SemanticNode(
                node_id=node_id,
                title=title,
                level=NodeLevel(level.lower()),
                page_index=page,
                summary=summary,
                parent_id=parent_id,
                children=[],
            )
            
            embedding_provider = get_embedding_provider()
            query_embedding = await embedding_provider.embed_single(summary or title)

            if parent_id:
                parent = next((n for n in nodes if n.node_id == parent_id), None)
                if parent:
                    parent.children.append(node_id)
            
            nodes.append(node)
            parent_stack.append((node_id, level))
        
        root_node_id = nodes[0].node_id if nodes else "0000"
        
        page_index = PageIndex(
            doc_id=doc_id,
            source_filename=source_filename,
            created_at=datetime.now().isoformat(),
            nodes=nodes,
            root_node_id=root_node_id,
            page_count=len(pages_data),
            node_count=len(nodes),
            language=language,
        )
        
        logger.info(f"[BUILD] ✓ Built tree with {len(nodes)} nodes (all LLM-generated)")
        save_chunks_index(nodes, doc_id)

        return page_index
    
    def _prepare_full_text(self, pages_data: List[Dict[str, Any]]) -> str:
        """Combine all pages into document text"""
        full_text = ""
        for page_data in pages_data:
            page_num = page_data.get("page", 1)
            text = page_data.get("clean_text", "")
            full_text += f"\n[PAGE {page_num}]\n{text}\n"
        return full_text
    
    async def _analyze_structure_with_llm(
        self,
        document_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Send document to Gemini for structure analysis.
        
        Gemini returns JSON with sections:
        {
            "sections": [
                {"title": "...", "level": "chapter", "page_index": 1, "content": "..."},
                {"title": "...", "level": "section", "page_index": 2, "content": "..."}
            ]
        }
        """
        
        text_to_send = document_text[:100000]
        prompt = DOCUMENT_STRUCTURE_ANALYSIS_PROMPT.format(
            document_text=text_to_send
        )
        
        try:
            response = await self.summarizer.analyze_document_structure(prompt)
            
            try:
                cleaned_response = self._sanitize_llm_json(response)
                data = json.loads(cleaned_response)
                sections = data.get("sections", [])
            except json.JSONDecodeError as e:
                logger.error(f"[LLM] Invalid JSON response: {response}")
                raise ValueError("LLM returned invalid JSON") from e
            
            if not sections:
                logger.warning("[LLM] Gemini returned no sections")
                return []
            
            logger.info(f"[LLM] Gemini returned {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"[LLM] Structure analysis failed: {e}")
            raise
    
    def _get_parent_level(self, level: str) -> Optional[str]:
        """Get parent level for a given level"""
        hierarchy = ["chapter", "section", "subsection", "paragraph"]
        if level not in hierarchy:
            return None
        idx = hierarchy.index(level)
        return hierarchy[idx - 1] if idx > 0 else None

    def _sanitize_llm_json(self, response: str) -> str:
        if not response:
            return ""

        text = response.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]

        return text

def save_semantic_tree_to_file(
    page_index: PageIndex,
    output_dir: str,
) -> str:
    """Save PageIndex to page_index.json"""
    output_path = Path(output_dir) / "page_index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_page_index(page_index, str(output_path))
    ogger.info(f"[SAVE] Saved PageIndex to {output_path}")
    return str(output_path)
