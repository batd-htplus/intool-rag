"""
INTEGRATION EXAMPLE - Complete Workflow
========================================

Ví dụ hoàn chỉnh: Upload PDF → Build structure → Query → Get answer

Yêu cầu:
- FastAPI endpoint nhận PDF
- Xây dựng semantic tree
- Chunking
- Query với semantic awareness
"""

import asyncio
from pathlib import Path
from typing import Optional

# Build Phase
from rag.ingest.semantic_tree_builder import SemanticTreeBuilder, save_semantic_tree_to_file
from rag.ingest.node_aware_chunker import ChunksBuilder
from rag.ingest.schemas import load_page_index, load_chunks_index, save_chunks_index

# Query Phase
from rag.query.query_retriever import QueryTimeRetriever
from rag.query.context_builder import ContextBuilder

# Services
from rag.llm.embedding_service import get_embedding_service
from rag.llm.llm_service import get_llm_service
from rag.ingest.ocr.pdf_ocr import PDFOCREngine


class RAGPipeline:
    """
    Complete RAG pipeline with semantic structure awareness.
    """
    
    def __init__(self, data_dir: str = "data/rag"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.tree_builder = SemanticTreeBuilder()
        self.chunks_builder = ChunksBuilder()
        self.query_retriever = QueryTimeRetriever()
        self.context_builder = ContextBuilder()
        
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()
    
    # ========================================================================
    # PHASE 1: BUILD (OFFLINE)
    # ========================================================================
    
    async def build_document(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
    ) -> dict:
        """
        Build semantic structure for a PDF document.
        
        Steps:
        1. Extract text from PDF (OCR if needed)
        2. Analyze structure → generate semantic tree
        3. Chunk text deterministically
        4. Embed chunks
        5. Save artifacts
        
        Returns:
        {
            "doc_id": "attention-paper",
            "status": "success",
            "page_count": 13,
            "node_count": 25,
            "chunk_count": 150,
            "output_dir": "data/rag/attention-paper"
        }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = pdf_path.stem.replace(" ", "-").lower()
        
        output_dir = self.data_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[BUILD] Starting build for {doc_id}")
        
        # Step 1: Extract text from PDF
        print(f"[BUILD] Step 1: Extracting text from {pdf_path}")
        ocr_engine = PDFOCREngine()
        pages_data = await ocr_engine.extract_pages(str(pdf_path))
        page_count = len(pages_data)
        print(f"[BUILD] Extracted {page_count} pages")
        
        # Step 2: Build semantic tree
        print(f"[BUILD] Step 2: Building semantic tree (LLM)")
        page_index = await self.tree_builder.build_tree(
            doc_id=doc_id,
            source_filename=pdf_path.name,
            pages_data=pages_data,
            language="en",
        )
        print(f"[BUILD] Created {len(page_index.nodes)} semantic nodes")
        
        # Save page_index
        page_index_path = save_semantic_tree_to_file(page_index, str(output_dir))
        
        # Step 3: Build chunks
        print(f"[BUILD] Step 3: Chunking text")
        chunks_index = await self.chunks_builder.build_chunks(
            doc_id=doc_id,
            page_index=page_index,
            pages_data=pages_data,
        )
        chunk_count = len(chunks_index.chunks)
        print(f"[BUILD] Created {chunk_count} chunks")
        
        # Save chunks
        chunks_path = str(output_dir / "chunks.json")
        save_chunks_index(chunks_index, chunks_path)
        
        # Step 4: Embed chunks
        print(f"[BUILD] Step 4: Embedding chunks")
        await self._embed_chunks(
            doc_id=doc_id,
            chunks_index=chunks_index,
            output_dir=str(output_dir),
        )
        
        print(f"[BUILD] ✓ Successfully built {doc_id}")
        
        return {
            "doc_id": doc_id,
            "status": "success",
            "page_count": page_count,
            "node_count": len(page_index.nodes),
            "chunk_count": chunk_count,
            "output_dir": str(output_dir),
        }
    
    async def _embed_chunks(
        self,
        doc_id: str,
        chunks_index,
        output_dir: str,
    ) -> None:
        """Embed chunks and create FAISS index"""
        from rag.ingest.schemas import FAISSMeta
        from rag.storage.faiss_index import create_faiss_index, save_faiss_index
        
        chunk_texts = [c.text for c in chunks_index.chunks]
        
        # Embed in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            batch_embeddings = await self.embedding_service.embed_batch(batch)
            embeddings.extend(batch_embeddings)
            
            progress = min(i + batch_size, len(chunk_texts))
            print(f"[EMBED] {progress}/{len(chunk_texts)} chunks embedded")
        
        # Create FAISS index
        import numpy as np
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        faiss_index = create_faiss_index(embeddings_array)
        
        # Save index
        index_path = Path(output_dir) / f"{doc_id}.faiss"
        save_faiss_index(faiss_index, str(index_path))
        
        # Save metadata
        mappings = {}
        chunk_id_to_vector = {}
        for i, chunk in enumerate(chunks_index.chunks):
            chunk_id = chunk.chunk_id
            mappings[str(i)] = chunk_id
            chunk_id_to_vector[chunk_id] = str(i)
            chunk.embedding_id = f"vec_{i:04d}"
        
        # Update chunks with embedding_ids
        save_chunks_index(chunks_index, Path(output_dir) / "chunks.json")
        
        # Save metadata
        faiss_meta = FAISSMeta(
            doc_id=doc_id,
            created_at="",  # filled by __init__
            vector_count=len(embeddings),
            mappings=mappings,
            chunk_id_to_vector=chunk_id_to_vector,
        )
        
        from rag.ingest.schemas import save_faiss_meta
        save_faiss_meta(faiss_meta, Path(output_dir) / "faiss_meta.json")
        
        print(f"[EMBED] Saved FAISS index with {len(embeddings)} vectors")
    
    # ========================================================================
    # PHASE 2: QUERY (ONLINE)
    # ========================================================================
    
    async def query_document(
        self,
        doc_id: str,
        user_query: str,
    ) -> dict:
        """
        Answer a question about a document using semantic retrieval.
        
        Steps:
        1. Load semantic structure (page_index.json)
        2. Load chunks (chunks.json)
        3. Load FAISS index
        4. Embed user query
        5. Retrieve chunks with semantic awareness
        6. Group chunks by node
        7. Build context intelligently
        8. Generate answer with LLM
        9. Add citations
        
        Returns:
        {
            "answer": "The answer to the question...",
            "context": {
                "query_type": "simple",
                "block_count": 2,
                "estimated_tokens": 450,
                "blocks": [...]
            },
            "citations": [
                {"node_id": "0007", "title": "...", "page": 4}
            ]
        }
        """
        output_dir = self.data_dir / doc_id
        
        print(f"[QUERY] Processing query: {user_query[:50]}...")
        
        # Step 1: Load indices
        print(f"[QUERY] Loading semantic indices")
        page_index = load_page_index(str(output_dir / "page_index.json"))
        chunks_index = load_chunks_index(str(output_dir / "chunks.json"))
        
        # Load FAISS
        import faiss
        from rag.ingest.schemas import load_faiss_meta
        
        faiss_index = faiss.read_index(str(output_dir / f"{doc_id}.faiss"))
        faiss_meta = load_faiss_meta(str(output_dir / "faiss_meta.json"))
        
        # Step 2: Embed query
        print(f"[QUERY] Embedding query")
        query_vector = await self.embedding_service.embed(user_query)
        
        # Step 3: Retrieve chunks with semantic awareness
        print(f"[QUERY] Retrieving chunks from FAISS")
        result = await self.query_retriever.process_query(
            query_vector=query_vector,
            faiss_index=faiss_index,
            faiss_meta=faiss_meta,
            chunks_index=chunks_index,
            page_index=page_index,
            top_k=50,
        )
        
        primary_node_id = result["primary_node_id"]
        if not primary_node_id:
            return {
                "answer": "Unable to find relevant information",
                "context": {},
                "citations": [],
            }
        
        primary_node = page_index.get_node(primary_node_id)
        print(f"[QUERY] Selected primary node: {primary_node.title}")
        
        # Step 4: Build context with all chunks
        print(f"[QUERY] Building context")
        
        # Regroup chunks by node for context building
        node_chunks = {}
        for chunk in chunks_index.chunks:
            if chunk.node_id not in node_chunks:
                node_chunks[chunk.node_id] = []
            node_chunks[chunk.node_id].append(chunk)
        
        context_result = self.context_builder.build_context_adaptive(
            query=user_query,
            primary_node_id=primary_node_id,
            comparison_node_ids=result.get("all_nodes"),
            node_chunks=node_chunks,
            page_index=page_index,
        )
        
        # Step 5: Generate answer with LLM
        print(f"[QUERY] Generating answer with LLM")
        
        system_prompt = """You are a helpful AI assistant that answers questions about documents.
        
Instructions:
- Answer based ONLY on the provided context
- Be concise and accurate
- Cite the relevant sections when appropriate
- If information is not in the context, say "I cannot find this information in the document"
        """.strip()
        
        user_prompt = f"""Context from document:
{context_result['formatted_text']}

Question: {user_query}

Please provide a concise answer based on the context above.
        """.strip()
        
        answer = await self.llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        print(f"[QUERY] ✓ Generated answer")
        
        return {
            "answer": answer,
            "context": context_result,
            "citations": context_result["citations"],
            "primary_node": {
                "node_id": primary_node_id,
                "title": primary_node.title,
                "level": primary_node.level.value,
                "page": primary_node.page_index,
            },
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of the RAG pipeline"""
    
    pipeline = RAGPipeline()
    
    # Build a document
    print("=== BUILDING DOCUMENT ===")
    build_result = await pipeline.build_document(
        pdf_path="papers/attention.pdf",
        doc_id="attention-paper"
    )
    print(f"Build result: {build_result}")
    
    # Query the document
    print("\n=== QUERYING DOCUMENT ===")
    query_result = await pipeline.query_document(
        doc_id="attention-paper",
        user_query="What is the Transformer architecture?"
    )
    
    print(f"\nAnswer:")
    print(query_result["answer"])
    
    print(f"\nCitations:")
    for citation in query_result["citations"]:
        print(f"  - {citation['title']} (page {citation['page']})")
    
    print(f"\nContext used: {query_result['context']['block_count']} blocks")


if __name__ == "__main__":
    asyncio.run(main())
