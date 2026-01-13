from pydantic import BaseModel
from typing import Optional

class Message(BaseModel):
    role: str
    content: str

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    project: str

class ChatResponse(BaseModel):
    answer: str