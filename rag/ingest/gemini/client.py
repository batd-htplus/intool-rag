from google import genai
from typing import Optional

from rag.config import config

_client: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    """
    Lazily initialize and return a singleton Gemini client.
    """
    global _client

    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        _client = genai.Client(api_key=config.GEMINI_API_KEY)

    return _client
