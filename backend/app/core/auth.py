from fastapi import HTTPException, Depends, Header
from typing import Optional
from app.core.config import settings

async def verify_auth(authorization: Optional[str] = Header(None)):
    """
    Verify authentication token.
    In production, implement proper JWT validation.
    """
    if not settings.AUTH_ENABLED:
        return None
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # TODO: Implement JWT verification in production
    return authorization
