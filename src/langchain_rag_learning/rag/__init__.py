"""RAG (Retrieval-Augmented Generation) module for document processing and vector search."""

from .document_processor import DocumentProcessor
from .text_splitter import DocumentChunker, RecursiveCharacterTextSplitter, SemanticTextSplitter, ContextAwareTextSplitter  # Regular expressions for text processing
from .embeddings import EmbeddingManager, OpenAIEmbeddingProvider, HuggingFaceEmbeddingProvider, LocalEmbeddingProvider
from .vector_store import VectorStoreManager, ChromaVectorStore, FAISSVectorStore  # Regular expressions for text processing
from .retrievers import (  # Regular expressions for text processing
    BaseRetriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    Reranker,
    RetrievalEngine
)

__all__ = [
    'DocumentProcessor',
    'DocumentChunker',
    'RecursiveCharacterTextSplitter',
    'SemanticTextSplitter', 
    'ContextAwareTextSplitter',
    'EmbeddingManager',
    'OpenAIEmbeddingProvider',
    'HuggingFaceEmbeddingProvider',
    'LocalEmbeddingProvider',
    'VectorStoreManager',
    'ChromaVectorStore',
    'FAISSVectorStore',
    'BaseRetriever',
    'DenseRetriever',
    'SparseRetriever',
    'HybridRetriever',
    'Reranker',
    'RetrievalEngine'
]