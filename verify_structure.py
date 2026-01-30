
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.getcwd())

from unittest.mock import MagicMock
import sys

# Mock external dependencies
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["langchain_community.embeddings"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["pydantic"] = MagicMock()

async def verify_llm_factory():
    print("\n--- Verifying LLM Factory ---")
    try:
        from rag.llm.factory import get_llm
        print("Imported rag.llm.factory.get_llm")
        # Don't instantiate as it might require API keys, but check if we can import
    except Exception as e:
        print(f"FAILED to import rag.llm.factory: {e}")
        return False
    return True

async def verify_providers():
    print("\n--- Verifying Providers ---")
    try:
        from rag.providers.gemini.llm import GeminiLLM
        print("Imported rag.providers.gemini.llm.GeminiLLM")
    except ImportError as e:
        print(f"FAILED to import GeminiLLM: {e}")
        return False

    try:
        from rag.providers.ollama.llm import OllamaLLMProvider
        print("Imported rag.providers.ollama.llm.OllamaLLMProvider")
    except ImportError as e:
        print(f"FAILED to import OllamaLLMProvider: {e}")
        return False
        
    try:
        from rag.providers.hf.embeddings import HuggingFaceEmbeddingProvider
        print("Imported rag.providers.hf.embeddings.HuggingFaceEmbeddingProvider")
    except ImportError as e:
        print(f"FAILED to import HuggingFaceEmbeddingProvider: {e}")
        return False
    return True

async def verify_ingest_imports():
    print("\n--- Verifying Ingest Imports ---")
    try:
        from rag.ingest.semantic.tree_builder import SemanticTreeBuilder
        print("Imported rag.ingest.semantic.tree_builder.SemanticTreeBuilder")
    except ImportError as e:
        print(f"FAILED to import SemanticTreeBuilder: {e}")
        return False
        
    try:
        from rag.ingest.node_aware_chunker import NodeAwareChunker
        print("Imported rag.ingest.node_aware_chunker.NodeAwareChunker")
    except ImportError as e:
        print(f"FAILED to import NodeAwareChunker: {e}")
        return False
    return True

async def verify_router_imports():
    print("\n--- Verifying Router Imports ---")
    try:
        from rag.routers.page_aware_v2 import router
        print("Imported rag.routers.page_aware_v2.router")
    except ImportError as e:
        print(f"FAILED to import page_aware_v2 router: {e}")
        return False
    return True

async def main():
    print("Starting verification (Structure Refactor)...")
    ok = True
    ok &= await verify_llm_factory()
    ok &= await verify_providers()
    ok &= await verify_ingest_imports()
    ok &= await verify_router_imports()
    
    if ok:
        print("\nSUCCESS: All critical modules import correctly.")
    else:
        print("\nFAILURE: Some modules failed to import.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
