"""
Page Selection & Context Assembly

PAGE SELECTION (CORE AGENT LOGIC)
- Use PageIndex (structure) + semantic scores
- Select BEST page (not scattered chunks)
- Apply decision rules

CONTEXT ASSEMBLY
- Sort chunks by position in page
- Format context with page metadata
"""

from typing import Optional
from rag.agent.state import AgentState, SelectedPageState
from rag.agent.data_loader import AgentStorage
from rag.logging import logger

class PageSelector:
    """SELECT PAGE (CORE AGENT LOGIC)"""
    
    def __init__(self, storage: AgentStorage):
        self.storage = storage
    
    def select_page(
        self,
        state: AgentState,
    ) -> Optional[SelectedPageState]:
        """
        SELECT PAGE using PageIndex + semantic scores.
        
        Decision Rules:
        1️⃣ RULE 1: Cannot answer without selecting a page
        2️⃣ RULE 2: FAISS is candidate source, not ground truth
        3️⃣ RULE 3: PageIndex is authority for structure
        4️⃣ RULE 4: If page score too low → return None (cannot answer)
        
        Args:
            state: Agent state with page candidates
            
        Returns:
            Selected page or None if score too low
        """
        
        if not state.page_candidates:
            logger.warning("No page candidates available")
            return None
        
        best_candidate = state.page_candidates[0]
        
        logger.info(
            f"Best candidate: Page {best_candidate.page} "
            f"(score={best_candidate.score:.3f}, chunks={len(best_candidate.chunks)})"
        )
        
        MIN_PAGE_SCORE = 0.3
        if best_candidate.score < MIN_PAGE_SCORE:
            logger.warning(
                f"Page score {best_candidate.score:.3f} below threshold {MIN_PAGE_SCORE}"
            )
            return None
        
        page_info = self.storage.get_page_info(best_candidate.page)
        
        if not page_info:
            logger.warning(f"Page {best_candidate.page} not in PageIndex")
            return None
        
        selected = SelectedPageState(
            page=best_candidate.page,
            chapter=page_info.chapter,
            section=page_info.section,
            subsection=page_info.subsection,
            title=page_info.title,
            chunks=best_candidate.chunks,
            score=best_candidate.score,
        )
        
        logger.info(
            f"✓ Selected Page {selected.page}: "
            f"Chapter={selected.chapter}, Section={selected.section}, Title={selected.title}"
        )
        
        return selected


class ContextAssembler:
    """CONTEXT ASSEMBLY"""
    
    def __init__(self, storage: AgentStorage):
        self.storage = storage
    
    def assemble_context(
        self,
        selected_page: SelectedPageState,
        max_length: int = 8000,
    ) -> str:
        """
        Assemble context for LLM.
        
        Process:
        1. Sort chunks by position in page
        2. Build page metadata header
        3. Join chunk texts
        4. Limit to max_length
        
        Args:
            selected_page: Selected page with chunks
            max_length: Max context length
            
        Returns:
            Formatted context string
        """
        
        chunks = sorted(
            selected_page.chunks,
            key=lambda c: c.text[:50],
        )
        
        header_parts = []
        
        if selected_page.chapter:
            header_parts.append(f"Chapter {selected_page.chapter}")
        if selected_page.section:
            header_parts.append(f"Section {selected_page.section}")
        if selected_page.title:
            header_parts.append(selected_page.title)
        
        header = " › ".join(header_parts) if header_parts else f"Page {selected_page.page}"
        
        context_lines = [
            f"[{header}]",
            "",
        ]
        
        for chunk in chunks:
            context_lines.append(chunk.text)
            context_lines.append("")
        
        context = "\n".join(context_lines).strip()
        
        if len(context) > max_length:
            logger.warning(f"Context truncated: {len(context)} → {max_length}")
            context = context[:max_length] + "..."
        
        logger.info(f"Assembled context: {len(context)} chars")
        
        return context


async def select_page(
    state: AgentState,
    selector: PageSelector,
) -> None:
    """Execute: Page Selection"""
    
    selected = selector.select_page(state)
    
    if selected is None:
        logger.warning("⚠️  RULE 4: Cannot answer - page score too low")
        state.selected_page = None
    else:
        state.selected_page = selected


async def assemble_context(
    state: AgentState,
    assembler: ContextAssembler,
) -> None:
    """Execute: Context Assembly"""
    
    if not state.selected_page:
        logger.warning("⚠️  RULE 1: Cannot assemble context without selected page")
        state.context = ""
        return
    
    intent_config = state.get_intent_config()
    max_context_length = intent_config["max_context_length"]
    
    context = assembler.assemble_context(
        state.selected_page,
        max_length=max_context_length,
    )
    
    state.context = context
