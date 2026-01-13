import uuid
from typing import List, Dict, Any
from rag.ingest.loaders import load
from rag.ingest.chunker import chunk
from rag.vector_store.qdrant import upsert
from rag.logging import logger

_embedding_model = None

def get_embedding_model():
    """Get embedding model instance (lazy loaded, singleton)"""
    global _embedding_model
    if _embedding_model is None:
        from rag.embedding.http_client import HTTPEmbedding
        _embedding_model = HTTPEmbedding()
    return _embedding_model

class IngestPipeline:
    """Document ingestion pipeline"""
    
    async def ingest_document(
        self,
        filepath: str,
        project: str,
        language: str = "en",
        doc_id: str = None
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Load document
        2. Chunk text
        3. Embed chunks
        4. Upsert to vector DB
        """
        try:
            if not doc_id:
                doc_id = str(uuid.uuid4())
            
            logger.info(f"Starting ingestion: {filepath} -> {project}")
            
            documents = load(filepath)
            logger.info(f"Loaded {len(documents)} documents")
            
            total_chunks = 0
            
            for doc in documents:
                chunks = chunk(doc.text)
                logger.info(f"Document has {len(chunks)} chunks")
                
                if not chunks:
                    continue
                
                texts = []
                payloads = []
                ids = []
                
                for i, chunk_text in enumerate(chunks):
                    texts.append(chunk_text)
                    
                    payload = {
                        "project": project,
                        "source": filepath.split('/')[-1],
                        "language": language,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "text": chunk_text,
                        **doc.metadata
                    }
                    payloads.append(payload)
                    
                    ids.append(str(uuid.uuid4()))
                
                logger.info(f"Embedding {len(texts)} chunks...")
                from rag.config import config
                embedding_model = get_embedding_model()
                embeddings = embedding_model.embed(
                    texts,
                    instruction=config.EMBEDDING_PASSAGE_INSTRUCTION if config.EMBEDDING_PASSAGE_INSTRUCTION else None
                )
                
                logger.info(f"Upserting {len(ids)} vectors to Qdrant...")
                await upsert(embeddings, payloads, ids)
                
                total_chunks += len(chunks)
            
            logger.info(f"Ingestion completed: {total_chunks} chunks")
            
            return {
                "doc_id": doc_id,
                "chunks_created": total_chunks,
                "status": "completed"
            }
        
        except Exception as e:
            logger.error(f"Ingestion error: {str(e)}")
            raise

# Pipeline instance
pipeline = IngestPipeline()
