from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json
import time
import uuid
from typing import Optional
from app.schemas.chat import ChatRequest, ChatResponse, Message, ChatChoice
from app.services.rag_service import rag_service
from app.core.logging import logger
from app.core.auth import verify_auth

router = APIRouter(tags=["chat"])

@router.post("/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    token: Optional[str] = Depends(verify_auth)
):
    """
    OpenAI-compatible chat completions endpoint.
    Supports RAG-augmented responses with document filtering.
    """
    try:
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message in chat history")
        
        filters = request.filters or {}
        if request.project:
            filters["project"] = request.project
        
        rag_response = await rag_service.query(
            question=user_message,
            filters=filters,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_sources=True
        )
        
        response_text = rag_response.get("answer", "")
        sources = rag_response.get("sources", [])
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        if sources:
            sources_text = "\n\n**Sources:**\n"
            for i, source in enumerate(sources, 1):
                score = source.get("score", 0)
                metadata = source.get("metadata", {})
                doc_name = metadata.get("filename", "Document")
                sources_text += f"{i}. {doc_name} (relevance: {score:.2%})\n"
            response_text += sources_text
        
        return ChatResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split())
            }
        )
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    token: Optional[str] = Depends(verify_auth)
):
    """
    Streaming chat completions endpoint.
    Sends chunks as server-sent events.
    """
    async def generate():
        try:
            user_message = None
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_message = msg.content
                    break
            
            if not user_message:
                yield f"data: {json.dumps({'error': 'No user message'})}\n\n"
                return
            
            filters = request.filters or {}
            if request.project:
                filters["project"] = request.project
            
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            
            async for chunk in rag_service.query_stream(
                question=user_message,
                filters=filters,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            finish_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(finish_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
