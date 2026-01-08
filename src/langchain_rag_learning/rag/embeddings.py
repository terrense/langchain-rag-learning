"""Vector embedding module for document vectorization and similarity search."""

import asyncio  # Async programming support for concurrent operations
import hashlib
import logging  # Structured logging for debugging and monitoring
import time  # Time utilities for performance measurement
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union  # Type hints for better code documentation

import numpy as np  # Numerical computing library

from langchain_rag_learning.core.config import get_settings  # LangChain framework for LLM applications
from langchain_rag_learning.core.exceptions import DocumentProcessingError  # LangChain framework for LLM applications
from langchain_rag_learning.core.models import DocumentChunk  # LangChain framework for LLM applications

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: Name of the embedding model
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self._model = None
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass
    
    async def _load_model(self):
        """Load the embedding model if not already loaded."""
        if self._model is None:
            await self._initialize_model()
    
    @abstractmethod
    async def _initialize_model(self):
        """Initialize the embedding model."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using OpenAI API."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or get_settings().openai_api_key
        self.client = None
        
        # Model dimensions mapping
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
    
    async def _initialize_model(self):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            
            if not self.api_key:
                raise DocumentProcessingError("OpenAI API key not provided")
            
            self.client = AsyncOpenAI(api_key=self.api_key)
            self._model = True  # Mark as initialized
            
        except ImportError:
            raise DocumentProcessingError("OpenAI library not installed. Install with: pip install openai")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI API."""
        await self._load_model()
        
        try:
            # Process in batches to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Add small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to embed documents with OpenAI: {str(e)}")
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query using OpenAI API."""
        await self._load_model()
        
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to embed query with OpenAI: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model."""
        return self.model_dimensions.get(self.model_name, 1536)


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace embedding provider.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ('cpu', 'cuda', etc.)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.device = device
        self._embedding_dimension = None
    
    async def _initialize_model(self):
        """Initialize HuggingFace model."""
        try:
            from sentence_transformers import SentenceTransformer  # HuggingFace transformers for NLP models
            
            self._model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            test_embedding = self._model.encode(["test"])
            self._embedding_dimension = len(test_embedding[0])
            
        except ImportError:
            raise DocumentProcessingError(
                "sentence-transformers library not installed. Install with: pip install sentence-transformers"
            )
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using HuggingFace model."""
        await self._load_model()
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(texts, convert_to_numpy=True)
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to embed documents with HuggingFace: {str(e)}")
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query using HuggingFace model."""
        await self._load_model()
        
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode([text], convert_to_numpy=True)
            )
            
            return embedding[0].tolist()
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to embed query with HuggingFace: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dimension is None:
            # Common dimensions for popular models
            dimension_map = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-distilroberta-v1": 768,
                "paraphrase-MiniLM-L6-v2": 384,
                "paraphrase-mpnet-base-v2": 768
            }
            return dimension_map.get(self.model_name, 384)
        
        return self._embedding_dimension


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using transformers library directly."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize local embedding provider.
        
        Args:
            model_name: Model name or path
            device: Device to run model on
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.device = device or ("cuda" if self._is_cuda_available() else "cpu")
        self.tokenizer = None
        self._embedding_dimension = None
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch  # PyTorch for deep learning
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def _initialize_model(self):
        """Initialize local model."""
        try:
            from transformers import AutoTokenizer, AutoModel  # HuggingFace transformers for NLP models
            import torch  # PyTorch for deep learning
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer(
                    ["test"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self._model(**test_input)
                # Use mean pooling
                embeddings = self._mean_pooling(outputs, test_input['attention_mask'])
                self._embedding_dimension = embeddings.shape[1]
            
        except ImportError:
            raise DocumentProcessingError(
                "transformers library not installed. Install with: pip install transformers torch"
            )
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        import torch  # PyTorch for deep learning
        
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using local model."""
        await self._load_model()
        
        try:
            import torch  # PyTorch for deep learning
            
            loop = asyncio.get_event_loop()
            
            def _embed_batch(batch_texts):
                """
                 Embed Batch function implementation.
                """
                with torch.no_grad():
                    encoded_input = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    model_output = self._model(**encoded_input)
                    embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    return embeddings.cpu().numpy()
            
            # Process in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await loop.run_in_executor(None, _embed_batch, batch)
                all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to embed documents with local model: {str(e)}")
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed query using local model."""
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dimension or 384


class EmbeddingCache:
    """Cache for embedding vectors to avoid recomputation."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._get_cache_key(text, model_name)
        
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        
        return None
    
    def put(self, text: str, model_name: str, embedding: List[float]):
        """Put embedding in cache."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._get_cache_key(text, model_name)
        self.cache[key] = embedding
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Evict oldest accessed item from cache."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class EmbeddingManager:
    """Main embedding manager that coordinates different providers and caching."""
    
    def __init__(self, cache_size: int = 10000):
        """
        Initialize embedding manager.
        
        Args:
            cache_size: Size of embedding cache
        """
        self.providers = {}
        self.cache = EmbeddingCache(cache_size)
        self.default_provider = None
        
        # Initialize default providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available embedding providers."""
        settings = get_settings()
        
        # OpenAI provider
        if settings.openai_api_key:
            self.providers['openai'] = OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key
            )
            if not self.default_provider:
                self.default_provider = 'openai'
        
        # HuggingFace provider (always available)
        self.providers['huggingface'] = HuggingFaceEmbeddingProvider()
        if not self.default_provider:
            self.default_provider = 'huggingface'
        
        # Local provider
        self.providers['local'] = LocalEmbeddingProvider()
    
    def add_provider(self, name: str, provider: BaseEmbeddingProvider):
        """Add a custom embedding provider."""
        self.providers[name] = provider
        
        if not self.default_provider:
            self.default_provider = name
    
    def set_default_provider(self, provider_name: str):
        """Set the default embedding provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        self.default_provider = provider_name
    
    async def embed_documents(
        self,
        texts: List[str],
        provider: Optional[str] = None,
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            provider: Embedding provider to use
            use_cache: Whether to use caching
            
        Returns:
            List of embedding vectors
        """
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        embedding_provider = self.providers[provider_name]
        
        if not use_cache:
            return await embedding_provider.embed_documents(texts)
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text, embedding_provider.model_name)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = await embedding_provider.embed_documents(uncached_texts)
            
            # Cache new embeddings and update result
            for idx, embedding in zip(uncached_indices, new_embeddings):
                self.cache.put(texts[idx], embedding_provider.model_name, embedding)
                embeddings[idx] = embedding
        
        return embeddings
    
    async def embed_query(
        self,
        text: str,
        provider: Optional[str] = None,
        use_cache: bool = True
    ) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed
            provider: Embedding provider to use
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector
        """
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        embedding_provider = self.providers[provider_name]
        
        if use_cache:
            cached_embedding = self.cache.get(text, embedding_provider.model_name)
            if cached_embedding:
                return cached_embedding
        
        embedding = await embedding_provider.embed_query(text)
        
        if use_cache:
            self.cache.put(text, embedding_provider.model_name, embedding)
        
        return embedding
    
    async def embed_document_chunks(
        self,
        chunks: List[DocumentChunk],
        provider: Optional[str] = None,
        use_cache: bool = True,
        batch_size: int = 100
    ) -> List[DocumentChunk]:
        """
        Embed document chunks and update them with embeddings.
        
        Args:
            chunks: List of DocumentChunk objects
            provider: Embedding provider to use
            use_cache: Whether to use caching
            batch_size: Batch size for processing
            
        Returns:
            Updated DocumentChunk objects with embeddings
        """
        if not chunks:
            return chunks
        
        # Process in batches
        updated_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            # Get embeddings for batch
            embeddings = await self.embed_documents(
                batch_texts,
                provider=provider,
                use_cache=use_cache
            )
            
            # Update chunks with embeddings
            for chunk, embedding in zip(batch_chunks, embeddings):
                chunk.embedding = embedding
                
                # Add embedding metadata
                if 'embedding_info' not in chunk.metadata:
                    chunk.metadata['embedding_info'] = {}
                
                chunk.metadata['embedding_info'].update({
                    'provider': provider or self.default_provider,
                    'model': self.providers[provider or self.default_provider].model_name,
                    'dimension': len(embedding),
                    'embedded_at': time.time()
                })
            
            updated_chunks.extend(batch_chunks)
            
            # Add small delay between batches
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.1)
        
        logger.info(f"Embedded {len(chunks)} document chunks")
        return updated_chunks
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")
        
        provider = self.providers[provider_name]
        
        return {
            'name': provider_name,
            'model_name': provider.model_name,
            'embedding_dimension': provider.get_embedding_dimension(),
            'class': provider.__class__.__name__
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': self.cache.size(),
            'max_size': self.cache.max_size,
            'hit_rate': getattr(self.cache, 'hit_rate', 0.0)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()


# Utility functions for similarity calculations
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    
    return np.linalg.norm(a_np - b_np)


def batch_cosine_similarity(
    query_embedding: List[float],
    document_embeddings: List[List[float]]
) -> List[float]:
    """Calculate cosine similarity between query and multiple documents."""
    query_np = np.array(query_embedding)
    docs_np = np.array(document_embeddings)
    
    # Normalize vectors
    query_norm = query_np / np.linalg.norm(query_np)
    docs_norm = docs_np / np.linalg.norm(docs_np, axis=1, keepdims=True)
    
    # Calculate similarities
    similarities = np.dot(docs_norm, query_norm)
    
    return similarities.tolist()