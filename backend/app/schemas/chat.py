from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatRequest(BaseModel):
    model: str = "AI-HTPv.10"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    project: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]