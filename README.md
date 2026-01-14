# RAG System

Retrieval Augmented Generation with hybrid search, semantic chunking and multi-service architecture.


## 🔑 Core Features

### 1. Hybrid Search
- **Vector search** (BGE-M3 embeddings) + **BM25 keyword search**
- Optional reranking with cross-encoder
- Dynamic weighting: `VECTOR_WEIGHT=0.7`, `BM25_WEIGHT=0.3`

### 2. Semantic Chunking
- Chunk by documents (Markdown headings, sections)
- Token-aware for BGE-M3 (~512 tokens/chunk)
- CJK-aware (character-based for JP/)

### 3. Multi-Level Caching
- **Embedding cache**: Re-embedding text duplicate
- **Query cache**: Cache retrieval results
- File-based with content hash

### 4. Provider Abstraction
- Switch between Ollama, HuggingFace, OpenAI
- HTTP-based providers with connection pooling
- Dependency Injection container

**Last Updated**: 2025-01-14  
**Version**: 0.2.0
