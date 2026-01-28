"""
RAG Agent API Integration
=========================

FastAPI endpoints for the Page-Aware Agent.

Endpoints:
- POST /agent/query - Query agent
- GET /agent/health - Health check
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from rag.agent import PageAwareAgent
from rag.logging import logger

router = APIRouter(prefix="/agent", tags=["agent"])

# Request/Response models

class AgentQueryRequest(BaseModel):
    """Agent query request"""
    question: str
    project: Optional[str] = None


class AgentSource(BaseModel):
    """Citation source"""
    page: int
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    title: Optional[str] = None


class AgentQueryResponse(BaseModel):
    """Agent query response"""
    answer: str
    source: Optional[AgentSource] = None


# Global agent instance (lazy loaded)
_agent: Optional[PageAwareAgent] = None


def get_agent() -> PageAwareAgent:
    """Get or initialize agent"""
    global _agent
    
    if _agent is None:
        # Configure paths (from config or environment)
        data_dir = "/path/to/data"  # TODO: configure
        faiss_index_path = "/path/to/faiss.index"  # TODO: configure
        
        _agent = PageAwareAgent(
            data_dir=data_dir,
            faiss_index_path=faiss_index_path,
            llm_model="gpt-4-turbo",
            embeddings_model="text-embedding-3-small",
        )
    
    return _agent


# Endpoints

@router.post("/query", response_model=AgentQueryResponse)
async def query(request: AgentQueryRequest) -> AgentQueryResponse:
    """
    Query the Page-Aware Agent.
    
    Pipeline:
    1. Query ingestion
    2. Query normalization
    3. Intent classification
    4. Semantic search (FAISS)
    5. Load chunk content
    6. Group by page
    7. Page selection
    8. Context assembly
    9. Answer generation (LLM)
    10. Answer validation
    11. Response formatting
    
    Args:
        request: Query request with question
        
    Returns:
        Answer with source citation
    """
    try:
        agent = get_agent()
        
        logger.info(f"Agent query: {request.question}")
        
        # Execute query
        response = await agent.query(request.question)
        
        # Format response
        return AgentQueryResponse(
            answer=response["answer"],
            source=AgentSource(**response["source"]) if response.get("source") else None,
        )
    
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Health check"""
    try:
        agent = get_agent()
        return {
            "status": "ok",
            "system": "page-aware-agent",
            "agent_ready": agent is not None,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# Example usage in main app

"""
from fastapi import FastAPI
from rag.agent.api import router as agent_router

app = FastAPI(title="RAG Page-Aware Agent")
app.include_router(agent_router)

# Test:
# POST /agent/query
# {
#     "question": "What are embedding models?",
#     "project": "my_project"
# }
#
# Response:
# {
#     "answer": "Embedding models are neural networks that...",
#     "source": {
#         "page": 12,
#         "chapter": "3",
#         "section": "3.2",
#         "subsection": "3.2.1",
#         "title": "Embedding Models"
#     }
# }
"""
