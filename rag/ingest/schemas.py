from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import json
from datetime import datetime

class NodeLevel(str, Enum):
    """Hierarchical levels in document tree"""
    ROOT = "root"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"


@dataclass
class SemanticNode:
    """
    Represents a semantic unit in document hierarchy.
    
    Core fields:
    - node_id: Unique identifier (format: "0000" padded to 4 digits)
    - title: Section title/heading
    - level: Hierarchical level (chapter, section, subsection, paragraph)
    - page_index: Starting page number (1-based)
    
    Semantic fields:
    - summary: Concise summary of THIS node only (NOT parent context)
               
    Structure fields:
    - parent_id: Parent node ID (None if root)
    - children: List of child node IDs (empty if leaf)
    
    Metadata:
    - char_start, char_end: Character offsets in original text
    - token_estimate: Rough token count (for LLM planning)
    """
    
    node_id: str  # "0000", "0001", etc.
    title: str
    level: NodeLevel
    page_index: int
    
    summary: str = ""  # LLM-generated summary of THIS node
    
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticNode":
        """Create from dictionary"""
        data["level"] = NodeLevel(data["level"]) if isinstance(data["level"], str) else data["level"]
        data["children"] = data.get("children", [])
        return cls(**data)
    
    def is_leaf(self) -> bool:
        """Check if node has no children"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if node is root"""
        return self.parent_id is None


@dataclass
class PageIndex:
    """
    Complete page index for a document.
    
    Structure:
    - doc_id: Unique document identifier
    - source_filename: Original PDF filename
    - created_at: ISO 8601 timestamp
    - nodes: List of all semantic nodes (flat representation)
    - root_node_id: ID of root node (entry point for traversal)
    
    Metadata:
    - page_count: Total pages in document
    - node_count: Total semantic nodes
    - language: Document language code
    """
    
    doc_id: str
    source_filename: str
    created_at: str
    nodes: List[SemanticNode]
    root_node_id: str
    
    page_count: int = 0
    node_count: int = 0
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "created_at": self.created_at,
            "root_node_id": self.root_node_id,
            "page_count": self.page_count,
            "node_count": self.node_count,
            "language": self.language,
            "nodes": [node.to_dict() for node in self.nodes],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageIndex":
        """Create from dictionary"""
        nodes = [SemanticNode.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            doc_id=data["doc_id"],
            source_filename=data["source_filename"],
            created_at=data["created_at"],
            nodes=nodes,
            root_node_id=data["root_node_id"],
            page_count=data.get("page_count", 0),
            node_count=data.get("node_count", len(nodes)),
            language=data.get("language", "en"),
        )
    
    def get_node(self, node_id: str) -> Optional[SemanticNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def get_children(self, node_id: str) -> List[SemanticNode]:
        """Get direct children of a node"""
        node = self.get_node(node_id)
        if not node:
            return []
        children = []
        for child_id in node.children:
            child = self.get_node(child_id)
            if child:
                children.append(child)
        return children
    
    def get_parent(self, node_id: str) -> Optional[SemanticNode]:
        """Get parent of a node"""
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return None
        return self.get_node(node.parent_id)

@dataclass
class Chunk:
    """
    Atomic text unit for embedding (SOURCE OF TRUTH FOR TEXT).
    
    Core fields:
    - chunk_id: Unique identifier (format: "c_{page}_{index}")
    - node_id: Parent semantic node ID
    - page: Page number (1-based)
    - text: Raw text content (NEVER summarized)
    
    Positioning:
    - char_start: Character offset in node text
    - char_end: Character offset in node text
    - seq_index: Sequence index within node (for ordering)
    
    Metadata:
    - token_estimate: Rough token count
    - embedding_id: Reference to embedding vector ID (set during embedding phase)
    """
    
    chunk_id: str  # "c_004_002"
    node_id: str  # Parent node
    page: int
    text: str
    
    char_start: int = 0
    char_end: int = 0
    seq_index: int = 0
    
    token_estimate: int = 0
    embedding_id: Optional[str] = None  # Set during embedding phase
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ChunksIndex:
    """
    Collection of all chunks for a document.
    
    Fields:
    - doc_id: Document ID (must match PageIndex)
    - created_at: ISO 8601 timestamp
    - chunks: List of all chunks
    - chunk_count: Total number of chunks
    """
    
    doc_id: str
    created_at: str
    chunks: List[Chunk]
    chunk_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "created_at": self.created_at,
            "chunk_count": len(self.chunks),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunksIndex":
        """Create from dictionary"""
        chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
        return cls(
            doc_id=data["doc_id"],
            created_at=data["created_at"],
            chunks=chunks,
            chunk_count=data.get("chunk_count", len(chunks)),
        )
    
    def get_chunks_by_node(self, node_id: str) -> List[Chunk]:
        """Get all chunks belonging to a node"""
        return [c for c in self.chunks if c.node_id == node_id]
    
    def get_chunks_by_page(self, page: int) -> List[Chunk]:
        """Get all chunks on a page"""
        return [c for c in self.chunks if c.page == page]

@dataclass
class FAISSMeta:
    """
    Metadata mapping for FAISS vector index.
    
    Maps vector position → chunk_id.
    Used to retrieve chunk information after vector search.
    
    Fields:
    - doc_id: Document ID
    - created_at: ISO 8601 timestamp
    - vector_count: Number of vectors in index
    - mappings: Dict[str(vector_index) → chunk_id]
    - chunk_id_to_vector: Reverse mapping (optional, for convenience)
    """
    
    doc_id: str
    created_at: str
    vector_count: int
    mappings: Dict[str, str]  # "0": "c_004_002", "1": "c_004_003", ...
    chunk_id_to_vector: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "created_at": self.created_at,
            "vector_count": self.vector_count,
            "mappings": self.mappings,
            "chunk_id_to_vector": self.chunk_id_to_vector,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FAISSMeta":
        """Create from dictionary"""
        return cls(
            doc_id=data["doc_id"],
            created_at=data["created_at"],
            vector_count=data["vector_count"],
            mappings=data["mappings"],
            chunk_id_to_vector=data.get("chunk_id_to_vector", {}),
        )
    
    def get_chunk_id(self, vector_index: int) -> Optional[str]:
        """Get chunk ID from vector index"""
        return self.mappings.get(str(vector_index))
    
    def get_vector_index(self, chunk_id: str) -> Optional[int]:
        """Get vector index from chunk ID"""
        vid = self.chunk_id_to_vector.get(chunk_id)
        if vid is not None:
            return int(vid)
        return None

def validate_node_id(node_id: str) -> bool:
    """Validate node ID format (4-digit padded)"""
    try:
        int(node_id)
        return len(node_id) == 4
    except ValueError:
        return False


def validate_chunk_id(chunk_id: str) -> bool:
    """Validate chunk ID format (c_{page}_{index})"""
    parts = chunk_id.split("_")
    if len(parts) != 3:
        return False
    if parts[0] != "c":
        return False
    try:
        int(parts[1])  # page
        int(parts[2])  # index
        return True
    except ValueError:
        return False


def save_page_index(page_index: PageIndex, filepath: str) -> None:
    """Save PageIndex to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(page_index.to_dict(), f, indent=2, ensure_ascii=False)


def load_page_index(filepath: str) -> PageIndex:
    """Load PageIndex from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PageIndex.from_dict(data)


def save_chunks_index(chunks: ChunksIndex, filepath: str) -> None:
    """Save ChunksIndex to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks.to_dict(), f, indent=2, ensure_ascii=False)


def load_chunks_index(filepath: str) -> ChunksIndex:
    """Load ChunksIndex from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ChunksIndex.from_dict(data)


def save_faiss_meta(meta: FAISSMeta, filepath: str) -> None:
    """Save FAISSMeta to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2)


def load_faiss_meta(filepath: str) -> FAISSMeta:
    """Load FAISSMeta from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FAISSMeta.from_dict(data)
