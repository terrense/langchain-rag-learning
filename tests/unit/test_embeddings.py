"""Unit tests for embeddings module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np

from langchain_rag_learning.core.models import DocumentChunk
from langchain_rag_learning.core.exceptions import DocumentProcessingError
from langchain_rag_learning.rag.embeddings import (
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    LocalEmbeddingProvider,
    EmbeddingCache,
    EmbeddingManager,
    cosine_similarity,
    euclidean_distance,
    batch_cosine_similarity
)


class TestOpenAIEmbeddingProvider:
    """Test cases for OpenAIEmbeddingProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create an OpenAI embedding provider for testing."""
        return OpenAIEmbeddingProvider(api_key="test_key")
    
    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.model_name == "text-embedding-3-small"
        assert provider.api_key == "test_key"
        assert provider.get_embedding_dimension() == 1536
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-large",
            api_key="test_key"
        )
        assert provider.model_name == "text-embedding-3-large"
        assert provider.get_embedding_dimension() == 3072
    
    @pytest.mark.asyncio
    async def test_initialize_model_missing_library(self, provider):
        """Test initialization with missing OpenAI library."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(DocumentProcessingError, match="OpenAI library not installed"):
                await provider._initialize_model()
    
    @pytest.mark.asyncio
    async def test_initialize_model_missing_api_key(self):
        """Test initialization without API key."""
        provider = OpenAIEmbeddingProvider(api_key=None)
        
        with patch('langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.openai_api_key = None
            
            with pytest.raises(DocumentProcessingError, match="OpenAI API key not provided"):
                await provider._initialize_model()
    
    @pytest.mark.asyncio
    async def test_embed_documents_success(self, provider):
        """Test successful document embedding."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        with patch('langchain_rag_learning.rag.embeddings.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_client
            
            texts = ["text 1", "text 2"]
            embeddings = await provider.embed_documents(texts)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            
            mock_client.embeddings.create.assert_called_once_with(
                model=provider.model_name,
                input=texts
            )
    
    @pytest.mark.asyncio
    async def test_embed_documents_batching(self, provider):
        """Test document embedding with batching."""
        mock_client = AsyncMock()
        
        # Mock responses for two batches
        mock_response_1 = Mock()
        mock_response_1.data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(100)]
        
        mock_response_2 = Mock()
        mock_response_2.data = [Mock(embedding=[0.4, 0.5, 0.6]) for _ in range(50)]
        
        mock_client.embeddings.create.side_effect = [mock_response_1, mock_response_2]
        
        with patch('langchain_rag_learning.rag.embeddings.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_client
            
            # Create 150 texts to trigger batching
            texts = [f"text {i}" for i in range(150)]
            embeddings = await provider.embed_documents(texts)
            
            assert len(embeddings) == 150
            assert mock_client.embeddings.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_embed_query_success(self, provider):
        """Test successful query embedding."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        with patch('langchain_rag_learning.rag.embeddings.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_client
            
            embedding = await provider.embed_query("test query")
            
            assert embedding == [0.1, 0.2, 0.3]
            mock_client.embeddings.create.assert_called_once_with(
                model=provider.model_name,
                input=["test query"]
            )
    
    @pytest.mark.asyncio
    async def test_embed_documents_api_error(self, provider):
        """Test handling of API errors."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        with patch('langchain_rag_learning.rag.embeddings.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_client
            
            with pytest.raises(DocumentProcessingError, match="Failed to embed documents with OpenAI"):
                await provider.embed_documents(["test"])


class TestHuggingFaceEmbeddingProvider:
    """Test cases for HuggingFaceEmbeddingProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a HuggingFace embedding provider for testing."""
        return HuggingFaceEmbeddingProvider()
    
    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.get_embedding_dimension() == 384
    
    @pytest.mark.asyncio
    async def test_initialize_model_missing_library(self, provider):
        """Test initialization with missing sentence-transformers library."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(DocumentProcessingError, match="sentence-transformers library not installed"):
                await provider._initialize_model()
    
    @pytest.mark.asyncio
    async def test_embed_documents_success(self, provider):
        """Test successful document embedding."""
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        
        with patch('langchain_rag_learning.rag.embeddings.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_model
            
            texts = ["text 1", "text 2"]
            embeddings = await provider.embed_documents(texts)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            
            mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)
    
    @pytest.mark.asyncio
    async def test_embed_query_success(self, provider):
        """Test successful query embedding."""
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embeddings
        
        with patch('langchain_rag_learning.rag.embeddings.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_model
            
            embedding = await provider.embed_query("test query")
            
            assert embedding == [0.1, 0.2, 0.3]
            mock_model.encode.assert_called_once_with(["test query"], convert_to_numpy=True)


class TestLocalEmbeddingProvider:
    """Test cases for LocalEmbeddingProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a local embedding provider for testing."""
        return LocalEmbeddingProvider()
    
    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert provider.get_embedding_dimension() == 384
    
    def test_is_cuda_available_true(self, provider):
        """Test CUDA availability detection when available."""
        with patch('torch.cuda.is_available', return_value=True):
            assert provider._is_cuda_available() == True
    
    def test_is_cuda_available_false(self, provider):
        """Test CUDA availability detection when not available."""
        with patch('torch.cuda.is_available', return_value=False):
            assert provider._is_cuda_available() == False
    
    def test_is_cuda_available_no_torch(self, provider):
        """Test CUDA availability detection without torch."""
        with patch('builtins.__import__', side_effect=ImportError):
            assert provider._is_cuda_available() == False
    
    @pytest.mark.asyncio
    async def test_initialize_model_missing_library(self, provider):
        """Test initialization with missing transformers library."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(DocumentProcessingError, match="transformers library not installed"):
                await provider._initialize_model()


class TestEmbeddingCache:
    """Test cases for EmbeddingCache class."""
    
    @pytest.fixture
    def cache(self):
        """Create an embedding cache for testing."""
        return EmbeddingCache(max_size=3)
    
    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 3
        assert cache.size() == 0
    
    def test_put_and_get(self, cache):
        """Test putting and getting embeddings."""
        embedding = [0.1, 0.2, 0.3]
        cache.put("test text", "model1", embedding)
        
        retrieved = cache.get("test text", "model1")
        assert retrieved == embedding
    
    def test_get_nonexistent(self, cache):
        """Test getting non-existent embedding."""
        result = cache.get("nonexistent", "model1")
        assert result is None
    
    def test_cache_eviction(self, cache):
        """Test cache eviction when max size is reached."""
        # Fill cache to max capacity
        cache.put("text1", "model1", [0.1, 0.2, 0.3])
        cache.put("text2", "model1", [0.4, 0.5, 0.6])
        cache.put("text3", "model1", [0.7, 0.8, 0.9])
        
        assert cache.size() == 3
        
        # Add one more item, should evict oldest
        cache.put("text4", "model1", [1.0, 1.1, 1.2])
        
        assert cache.size() == 3
        assert cache.get("text1", "model1") is None  # Should be evicted
        assert cache.get("text4", "model1") is not None  # Should be present
    
    def test_clear_cache(self, cache):
        """Test clearing the cache."""
        cache.put("text1", "model1", [0.1, 0.2, 0.3])
        cache.put("text2", "model1", [0.4, 0.5, 0.6])
        
        assert cache.size() == 2
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("text1", "model1") is None
        assert cache.get("text2", "model1") is None
    
    def test_different_models_different_keys(self, cache):
        """Test that different models create different cache keys."""
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]
        
        cache.put("same text", "model1", embedding1)
        cache.put("same text", "model2", embedding2)
        
        assert cache.get("same text", "model1") == embedding1
        assert cache.get("same text", "model2") == embedding2


class TestEmbeddingManager:
    """Test cases for EmbeddingManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create an embedding manager for testing."""
        with patch('langchain_rag_learning.core.config.get_settings') as mock_settings:
            mock_settings.return_value.openai_api_key = "test_key"
            return EmbeddingManager(cache_size=100)
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.providers) >= 2  # At least HuggingFace and local
        assert manager.default_provider is not None
        assert isinstance(manager.cache, EmbeddingCache)
    
    def test_add_provider(self, manager):
        """Test adding a custom provider."""
        custom_provider = Mock(spec=OpenAIEmbeddingProvider)
        manager.add_provider("custom", custom_provider)
        
        assert "custom" in manager.providers
        assert manager.providers["custom"] == custom_provider
    
    def test_set_default_provider(self, manager):
        """Test setting default provider."""
        manager.set_default_provider("huggingface")
        assert manager.default_provider == "huggingface"
    
    def test_set_invalid_default_provider(self, manager):
        """Test setting invalid default provider."""
        with pytest.raises(ValueError, match="Provider nonexistent not available"):
            manager.set_default_provider("nonexistent")
    
    @pytest.mark.asyncio
    async def test_embed_documents_with_cache(self, manager):
        """Test embedding documents with caching."""
        mock_provider = AsyncMock()
        mock_provider.model_name = "test_model"
        mock_provider.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        manager.providers["test"] = mock_provider
        manager.set_default_provider("test")
        
        texts = ["text1", "text2"]
        embeddings = await manager.embed_documents(texts, use_cache=True)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        
        # Second call should use cache
        embeddings2 = await manager.embed_documents(texts, use_cache=True)
        assert embeddings2 == embeddings
        
        # Provider should only be called once due to caching
        mock_provider.embed_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embed_documents_without_cache(self, manager):
        """Test embedding documents without caching."""
        mock_provider = AsyncMock()
        mock_provider.model_name = "test_model"
        mock_provider.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        manager.providers["test"] = mock_provider
        manager.set_default_provider("test")
        
        texts = ["text1"]
        await manager.embed_documents(texts, use_cache=False)
        await manager.embed_documents(texts, use_cache=False)
        
        # Provider should be called twice without caching
        assert mock_provider.embed_documents.call_count == 2
    
    @pytest.mark.asyncio
    async def test_embed_query(self, manager):
        """Test embedding a single query."""
        mock_provider = AsyncMock()
        mock_provider.model_name = "test_model"
        mock_provider.embed_query.return_value = [0.1, 0.2, 0.3]
        
        manager.providers["test"] = mock_provider
        manager.set_default_provider("test")
        
        embedding = await manager.embed_query("test query")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_provider.embed_query.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_embed_document_chunks(self, manager):
        """Test embedding document chunks."""
        mock_provider = AsyncMock()
        mock_provider.model_name = "test_model"
        mock_provider.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        manager.providers["test"] = mock_provider
        manager.set_default_provider("test")
        
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="content 1",
                document_id="doc1",
                chunk_index=0,
                start_char=0,
                end_char=10
            ),
            DocumentChunk(
                id="chunk2",
                content="content 2",
                document_id="doc1",
                chunk_index=1,
                start_char=10,
                end_char=20
            )
        ]
        
        updated_chunks = await manager.embed_document_chunks(chunks)
        
        assert len(updated_chunks) == 2
        assert updated_chunks[0].embedding == [0.1, 0.2, 0.3]
        assert updated_chunks[1].embedding == [0.4, 0.5, 0.6]
        
        # Check metadata was added
        for chunk in updated_chunks:
            assert 'embedding_info' in chunk.metadata
            assert chunk.metadata['embedding_info']['provider'] == 'test'
            assert chunk.metadata['embedding_info']['model'] == 'test_model'
    
    def test_get_available_providers(self, manager):
        """Test getting available providers."""
        providers = manager.get_available_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
    
    def test_get_provider_info(self, manager):
        """Test getting provider information."""
        # Add a mock provider for testing
        mock_provider = Mock()
        mock_provider.model_name = "test_model"
        mock_provider.get_embedding_dimension.return_value = 384
        mock_provider.__class__.__name__ = "TestProvider"
        
        manager.providers["test"] = mock_provider
        
        info = manager.get_provider_info("test")
        
        assert info['name'] == 'test'
        assert info['model_name'] == 'test_model'
        assert info['embedding_dimension'] == 384
        assert info['class'] == 'TestProvider'
    
    def test_get_cache_stats(self, manager):
        """Test getting cache statistics."""
        stats = manager.get_cache_stats()
        
        assert 'size' in stats
        assert 'max_size' in stats
        assert stats['max_size'] == 100
    
    def test_clear_cache(self, manager):
        """Test clearing the cache."""
        # Add something to cache first
        manager.cache.put("test", "model", [0.1, 0.2, 0.3])
        assert manager.cache.size() > 0
        
        manager.clear_cache()
        assert manager.cache.size() == 0


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6  # Should be 0 (orthogonal)
        
        # Test identical vectors
        similarity = cosine_similarity(a, a)
        assert abs(similarity - 1.0) < 1e-6  # Should be 1 (identical)
    
    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(a, b)
        assert similarity == 0.0
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        distance = euclidean_distance(a, b)
        assert abs(distance - 1.0) < 1e-6
        
        # Test identical vectors
        distance = euclidean_distance(a, a)
        assert abs(distance - 0.0) < 1e-6
    
    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity calculation."""
        query = [1.0, 0.0, 0.0]
        documents = [
            [1.0, 0.0, 0.0],  # Same as query
            [0.0, 1.0, 0.0],  # Orthogonal to query
            [-1.0, 0.0, 0.0]  # Opposite to query
        ]
        
        similarities = batch_cosine_similarity(query, documents)
        
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6   # Same direction
        assert abs(similarities[1] - 0.0) < 1e-6   # Orthogonal
        assert abs(similarities[2] - (-1.0)) < 1e-6  # Opposite direction


class TestIntegration:
    """Integration tests for embedding workflow."""
    
    @pytest.mark.asyncio
    async def test_full_embedding_workflow(self):
        """Test the complete embedding workflow."""
        # Create mock provider
        mock_provider = AsyncMock()
        mock_provider.model_name = "test_model"
        mock_provider.get_embedding_dimension.return_value = 3
        mock_provider.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_provider.embed_query.return_value = [0.7, 0.8, 0.9]
        
        # Create manager and add provider
        manager = EmbeddingManager(cache_size=10)
        manager.add_provider("test", mock_provider)
        manager.set_default_provider("test")
        
        # Create document chunks
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="First chunk content",
                document_id="doc1",
                chunk_index=0,
                start_char=0,
                end_char=19
            ),
            DocumentChunk(
                id="chunk2",
                content="Second chunk content",
                document_id="doc1",
                chunk_index=1,
                start_char=19,
                end_char=39
            )
        ]
        
        # Embed chunks
        embedded_chunks = await manager.embed_document_chunks(chunks)
        
        # Verify embeddings were added
        assert len(embedded_chunks) == 2
        assert embedded_chunks[0].embedding == [0.1, 0.2, 0.3]
        assert embedded_chunks[1].embedding == [0.4, 0.5, 0.6]
        
        # Verify metadata was added
        for chunk in embedded_chunks:
            assert 'embedding_info' in chunk.metadata
            assert chunk.metadata['embedding_info']['provider'] == 'test'
            assert chunk.metadata['embedding_info']['model'] == 'test_model'
            assert chunk.metadata['embedding_info']['dimension'] == 3
        
        # Test query embedding
        query_embedding = await manager.embed_query("test query")
        assert query_embedding == [0.7, 0.8, 0.9]
        
        # Test caching - second call should use cache
        query_embedding2 = await manager.embed_query("test query")
        assert query_embedding2 == query_embedding
        
        # Provider should only be called once for the query due to caching
        mock_provider.embed_query.assert_called_once()