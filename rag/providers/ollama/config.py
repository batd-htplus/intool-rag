from dataclasses import dataclass
from rag.config import config

@dataclass(frozen=True)
class OllamaSemanticConfig:
    base_url: str = config.LLM_BASE_URL
    model: str = config.LLM_MODEL
    temperature: float = config.LLM_TEMPERATURE
    max_tokens: int = config.LLM_MAX_TOKENS
    timeout: int = config.LLM_TIMEOUT
    num_ctx: int = 16384