"""RAG (Retrieval-Augmented Generation) module for document processing and vector search."""

from .document_processor import DocumentProcessor
from .text_splitter import DocumentChunker, RecursiveCharacterTextSplitter, SemanticTextSplitter, ContextAwareTextSplitter
from .embeddings import EmbeddingManager, OpenAIEmbeddingProvider, HuggingFaceEmbeddingProvider, LocalEmbeddingProvider
from .vector_store import VectorStoreManager, ChromaVectorStore, FAISSVectorStore
from .retrievers import (
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