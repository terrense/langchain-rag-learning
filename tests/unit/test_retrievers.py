"""Unit tests for retrieval engines."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_rag_learning.core.models import DocumentChunk
from langchain_rag_learning.rag.retrievers import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    Reranker,
    RetrievalEngine
)


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    chunks = []
    
    # Create test chunks with different content
    test_contents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Reinforcement learning trains agents through interaction with environments."
    ]
    
    for i, content in enumerate(test_contents):
        chunk = DocumentChunk(
            id=f"chunk_{i}",
            content=content,
            document_id=f"doc_{i // 2}",  # 2-3 chunks per document
            chunk_index=i,
            start_char=0,
            end_char=len(content),
            embedding=[0.1 * j for j in range(384)],  # Mock embedding
            metadata={"topic": "AI" if i < 3 else "ML"}
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    mock_store = AsyncMock()
    mock_store.similarity_search = AsyncMock()
    return mock_store


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    mock_manager = AsyncMock()
    mock_manager.embed_query = AsyncMock(return_value=[0.1] * 384)
    return mock_manager


class TestDenseRetriever:
    """Test dense retriever functionality."""
    
    @pytest.mark.asyncio
    async def test_dense_retrieval_basic(self, mock_vector_store, mock_embedding_manager, sample_chunks):
        """Test basic dense retrieval."""
        # Setup mock responses
        mock_vector_store.similarity_search.return_value = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.8),
            (sample_chunks[2], 0.7)
        ]
        
        # Create retriever
        retriever = DenseRetriever(mock_vector_store, mock_embedding_manager)
        
        # Test retrieval
        results = await retriever.retrieve("test query", k=3)
        
        # Verify results
        assert len(results) == 3
        assert results[0][1] == 0.9  # Check score
        assert results[0][0].id == "chunk_0"  # Check chunk
        
        # Verify mock calls
        mock_embedding_manager.embed_query.assert_called_once_with("test query", provider=None)
        mock_vector_store.similarity_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dense_retrieval_with_filters(self, mock_vector_store, mock_embedding_manager, sample_chunks):
        """Test dense retrieval with metadata filters."""
        mock_vector_store.similarity_search.return_value = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.8)
        ]
        
        retriever = DenseRetriever(mock_vector_store, mock_embedding_manager)
        
        filter_dict = {"topic": "AI"}
        results = await retriever.retrieve("test query", k=5, filter_dict=filter_dict)
        
        assert len(results) == 2
        mock_vector_store.similarity_search.assert_called_once_with(
            query_embedding=[0.1] * 384,
            k=5,
            filter_dict=filter_dict
        )
    
    @pytest.mark.asyncio
    async def test_dense_retrieval_similarity_threshold(self, mock_vector_store, mock_embedding_manager, sample_chunks):
        """Test dense retrieval with similarity threshold."""
        mock_vector_store.similarity_search.return_value = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.5),  # Below threshold
            (sample_chunks[2], 0.8)
        ]
        
        retriever = DenseRetriever(mock_vector_store, mock_embedding_manager)
        
        results = await retriever.retrieve("test query", k=5, similarity_threshold=0.7)
        
        # Should filter out chunk with score 0.5
        assert len(results) == 2
        assert all(score >= 0.7 for _, score in results)
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, mock_vector_store, mock_embedding_manager, sample_chunks):
        """Test query expansion functionality."""
        mock_vector_store.similarity_search.return_value = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.8),
            (sample_chunks[2], 0.7)
        ]
        
        retriever = DenseRetriever(mock_vector_store, mock_embedding_manager)
        
        expansion_terms = await retriever.get_query_expansion_terms("machine learning")
        
        assert isinstance(expansion_terms, list)
        # Should extract terms from retrieved content
        mock_vector_store.similarity_search.assert_called_once()


class TestSparseRetriever:
    """Test sparse retriever functionality."""
    
    def test_sparse_retriever_initialization(self, sample_chunks):
        """Test sparse retriever initialization and index building."""
        retriever = SparseRetriever(sample_chunks)
        
        # Check that index was built
        assert len(retriever.inverted_index) > 0
        assert len(retriever.doc_lengths) == len(sample_chunks)
        assert retriever.total_docs == len(sample_chunks)
        assert retriever.avg_doc_length > 0
    
    @pytest.mark.asyncio
    async def test_sparse_retrieval_bm25(self, sample_chunks):
        """Test BM25 scoring."""
        retriever = SparseRetriever(sample_chunks)
        
        results = await retriever.retrieve("machine learning", k=3, scoring_method="bm25")
        
        assert len(results) <= 3
        assert all(isinstance(score, float) for _, score in results)
        # Results should be sorted by score
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_sparse_retrieval_tfidf(self, sample_chunks):
        """Test TF-IDF scoring."""
        retriever = SparseRetriever(sample_chunks)
        
        results = await retriever.retrieve("neural networks", k=3, scoring_method="tfidf")
        
        assert len(results) <= 3
        assert all(isinstance(score, float) for _, score in results)
    
    @pytest.mark.asyncio
    async def test_sparse_retrieval_with_filters(self, sample_chunks):
        """Test sparse retrieval with metadata filters."""
        retriever = SparseRetriever(sample_chunks)
        
        filter_dict = {"topic": "AI"}
        results = await retriever.retrieve("learning", k=5, filter_dict=filter_dict)
        
        # Should only return chunks with topic "AI"
        for chunk, _ in results:
            assert chunk.metadata.get("topic") == "AI"
    
    @pytest.mark.asyncio
    async def test_query_expansion_sparse(self, sample_chunks):
        """Test query expansion in sparse retriever."""
        retriever = SparseRetriever(sample_chunks)
        
        results = await retriever.retrieve("machine", k=3, query_expansion=True)
        
        # Should work without errors
        assert isinstance(results, list)
    
    def test_tokenization(self, sample_chunks):
        """Test tokenization function."""
        retriever = SparseRetriever(sample_chunks)
        
        tokens = retriever._tokenize("Machine learning is great!")
        
        assert "machine" in tokens
        assert "learning" in tokens
        assert "great" in tokens
        # Should filter out short tokens and punctuation
        assert "is" not in tokens  # Too short
        assert "!" not in tokens  # Punctuation
    
    def test_update_index(self, sample_chunks):
        """Test index updating with new documents."""
        retriever = SparseRetriever(sample_chunks[:3])
        initial_doc_count = retriever.total_docs
        
        # Add new documents
        new_chunks = sample_chunks[3:]
        retriever.update_index(new_chunks)
        
        assert retriever.total_docs == initial_doc_count + len(new_chunks)
        assert len(retriever.documents) == len(sample_chunks)


class TestHybridRetriever:
    """Test hybrid retriever functionality."""
    
    @pytest.fixture
    def mock_dense_retriever(self, sample_chunks):
        """Create mock dense retriever."""
        mock_retriever = AsyncMock(spec=DenseRetriever)
        mock_retriever.retrieve = AsyncMock(return_value=[
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.8),
            (sample_chunks[2], 0.7)
        ])
        mock_retriever.get_name.return_value = "dense_test"
        return mock_retriever
    
    @pytest.fixture
    def mock_sparse_retriever(self, sample_chunks):
        """Create mock sparse retriever."""
        mock_retriever = AsyncMock(spec=SparseRetriever)
        mock_retriever.retrieve = AsyncMock(return_value=[
            (sample_chunks[1], 2.5),  # Different scoring scale
            (sample_chunks[3], 2.0),
            (sample_chunks[0], 1.5)
        ])
        mock_retriever.get_name.return_value = "sparse_test"
        return mock_retriever
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_rrf(self, mock_dense_retriever, mock_sparse_retriever):
        """Test hybrid retrieval with RRF fusion."""
        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever)
        
        results = await retriever.retrieve("test query", k=3, fusion_method="rrf")
        
        assert len(results) <= 3
        # Both retrievers should be called
        mock_dense_retriever.retrieve.assert_called_once()
        mock_sparse_retriever.retrieve.assert_called_once()
        
        # Results should have RRF scores
        for chunk, score in results:
            assert isinstance(score, float)
            assert score > 0
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_weighted(self, mock_dense_retriever, mock_sparse_retriever):
        """Test hybrid retrieval with weighted fusion."""
        retriever = HybridRetriever(
            mock_dense_retriever, 
            mock_sparse_retriever,
            dense_weight=0.7,
            sparse_weight=0.3
        )
        
        results = await retriever.retrieve("test query", k=3, fusion_method="weighted")
        
        assert len(results) <= 3
        # Check that weights are applied
        assert retriever.dense_weight == 0.7
        assert retriever.sparse_weight == 0.3
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_max_fusion(self, mock_dense_retriever, mock_sparse_retriever):
        """Test hybrid retrieval with max fusion."""
        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever)
        
        results = await retriever.retrieve("test query", k=3, fusion_method="max")
        
        assert len(results) <= 3
        # Should use max scores from either retriever
        for chunk, score in results:
            assert isinstance(score, float)
    
    def test_weight_validation(self, mock_dense_retriever, mock_sparse_retriever):
        """Test weight validation."""
        # Should work with valid weights
        retriever = HybridRetriever(
            mock_dense_retriever, 
            mock_sparse_retriever,
            dense_weight=0.6,
            sparse_weight=0.4
        )
        
        # Should warn about invalid weights but still work
        retriever = HybridRetriever(
            mock_dense_retriever, 
            mock_sparse_retriever,
            dense_weight=0.6,
            sparse_weight=0.5  # Doesn't sum to 1.0
        )
        
        # Update weights should validate
        with pytest.raises(ValueError):
            retriever.update_weights(0.6, 0.5)
    
    def test_score_normalization(self, mock_dense_retriever, mock_sparse_retriever, sample_chunks):
        """Test score normalization."""
        retriever = HybridRetriever(mock_dense_retriever, mock_sparse_retriever)
        
        # Test with different score ranges
        results = [
            (sample_chunks[0], 10.0),
            (sample_chunks[1], 5.0),
            (sample_chunks[2], 1.0)
        ]
        
        normalized = retriever._normalize_scores(results)
        
        # Should be normalized to [0, 1]
        scores = [score for _, score in normalized]
        assert max(scores) == 1.0
        assert min(scores) == 0.0
        
        # Test with identical scores
        identical_results = [(chunk, 5.0) for chunk, _ in results]
        normalized_identical = retriever._normalize_scores(identical_results)
        
        # All should be 1.0 when identical
        assert all(score == 1.0 for _, score in normalized_identical)


class TestReranker:
    """Test reranker functionality."""
    
    @pytest.mark.asyncio
    async def test_reranker_initialization(self):
        """Test reranker initialization."""
        reranker = Reranker()
        
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert not reranker._model_loaded
    
    @pytest.mark.asyncio
    async def test_reranker_fallback(self, sample_chunks):
        """Test reranker fallback when model loading fails."""
        reranker = Reranker()
        
        # Mock model loading failure
        with patch.object(reranker, '_load_model', side_effect=Exception("Model loading failed")):
            results = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.8)]
            
            reranked = await reranker.rerank("test query", results)
            
            # Should return original results
            assert reranked == results
    
    @pytest.mark.asyncio
    @patch('langchain_rag_learning.rag.retrievers.CrossEncoder')
    async def test_reranker_with_mock_model(self, mock_cross_encoder, sample_chunks):
        """Test reranker with mocked model."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.85, 0.75]
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker()
        
        results = [
            (sample_chunks[0], 0.7),
            (sample_chunks[1], 0.8),
            (sample_chunks[2], 0.9)
        ]
        
        reranked = await reranker.rerank("test query", results, top_k=2)
        
        # Should return top 2 reranked results
        assert len(reranked) == 2
        # Should be sorted by reranked scores (0.95, 0.85, 0.75)
        assert reranked[0][1] == 0.95
        assert reranked[1][1] == 0.85
    
    def test_diversity_penalty(self, sample_chunks):
        """Test diversity penalty calculation."""
        reranker = Reranker()
        
        results = [
            (sample_chunks[0], 0.9),
            (sample_chunks[1], 0.8),
            (sample_chunks[2], 0.7)
        ]
        
        penalized = reranker._apply_diversity_novelty_penalties(
            results, diversity_penalty=0.1, novelty_penalty=0.0
        )
        
        # Should apply penalties
        assert len(penalized) == len(results)
        # First result should have no penalty, others should have some penalty
        assert penalized[0][1] == 0.9  # No penalty for first
        assert penalized[1][1] < 0.8   # Some penalty for second
    
    def test_content_similarity(self, sample_chunks):
        """Test content similarity calculation."""
        reranker = Reranker()
        
        similarity = reranker._calculate_content_similarity(
            "machine learning algorithms",
            "machine learning models"
        )
        
        assert 0 <= similarity <= 1
        assert similarity > 0  # Should have some similarity
        
        # Test with identical content
        identical_similarity = reranker._calculate_content_similarity(
            "test content",
            "test content"
        )
        assert identical_similarity == 1.0
        
        # Test with completely different content
        different_similarity = reranker._calculate_content_similarity(
            "machine learning",
            "cooking recipes"
        )
        assert different_similarity < 0.5


class TestRetrievalEngine:
    """Test retrieval engine coordination."""
    
    @pytest.fixture
    def mock_vector_store_manager(self):
        """Create mock vector store manager."""
        return MagicMock()
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        return MagicMock()
    
    def test_retrieval_engine_initialization(self, mock_vector_store_manager, mock_embedding_manager):
        """Test retrieval engine initialization."""
        engine = RetrievalEngine(mock_vector_store_manager, mock_embedding_manager)
        
        assert engine.vector_store_manager == mock_vector_store_manager
        assert engine.embedding_manager == mock_embedding_manager
        assert engine.default_retriever == "hybrid"
        assert len(engine.retrievers) == 0
    
    def test_add_retrievers(self, mock_vector_store_manager, mock_embedding_manager, sample_chunks):
        """Test adding different types of retrievers."""
        engine = RetrievalEngine(mock_vector_store_manager, mock_embedding_manager)
        
        # Add dense retriever
        mock_vector_store = MagicMock()
        engine.add_dense_retriever("dense_test", mock_vector_store)
        
        assert "dense_test" in engine.retrievers
        assert isinstance(engine.retrievers["dense_test"], DenseRetriever)
        
        # Add sparse retriever
        engine.add_sparse_retriever("sparse_test", sample_chunks)
        
        assert "sparse_test" in engine.retrievers
        assert isinstance(engine.retrievers["sparse_test"], SparseRetriever)
        
        # Add hybrid retriever
        engine.add_hybrid_retriever("hybrid_test", "dense_test", "sparse_test")
        
        assert "hybrid_test" in engine.retrievers
        assert isinstance(engine.retrievers["hybrid_test"], HybridRetriever)
    
    def test_add_hybrid_retriever_validation(self, mock_vector_store_manager, mock_embedding_manager):
        """Test hybrid retriever validation."""
        engine = RetrievalEngine(mock_vector_store_manager, mock_embedding_manager)
        
        # Should fail with non-existent retrievers
        with pytest.raises(ValueError):
            engine.add_hybrid_retriever("hybrid_test", "nonexistent", "also_nonexistent")
    
    @pytest.mark.asyncio
    async def test_retrieve_with_engine(self, mock_vector_store_manager, mock_embedding_manager, sample_chunks):
        """Test retrieval through engine."""
        engine = RetrievalEngine(mock_vector_store_manager, mock_embedding_manager)
        
        # Add a sparse retriever for testing
        engine.add_sparse_retriever("test_retriever", sample_chunks)
        
        results = await engine.retrieve("machine learning", retriever_name="test_retriever", k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
    
    def test_get_retriever_info(self, mock_vector_store_manager, mock_embedding_manager, sample_chunks):
        """Test getting retriever information."""
        engine = RetrievalEngine(mock_vector_store_manager, mock_embedding_manager)
        
        engine.add_sparse_retriever("test_retriever", sample_chunks)
        
        info = engine.get_retriever_info("test_retriever")
        
        assert info["name"] == "test_retriever"
        assert info["type"] == "SparseRetriever"
        assert "document_count" in info
        assert "vocabulary_size" in info
    
    @pytest.mark.asyncio
    async def test_evaluate_retrieval(self, mock_vector_store_manager, mock_embedding_manager, sample_chunks):
        """Test retrieval evaluation."""
        engine = RetrievalEngine(mock_vector_store_manager, mock_embedding_manager)
        
        engine.add_sparse_retriever("test_retriever", sample_chunks)
        
        test_queries = ["machine learning", "neural networks"]
        ground_truth = [["chunk_0", "chunk_1"], ["chunk_1", "chunk_2"]]
        
        metrics = await engine.evaluate_retrieval(
            test_queries, ground_truth, retriever_name="test_retriever", k=3
        )
        
        assert "precision@k" in metrics
        assert "recall@k" in metrics
        assert "f1@k" in metrics
        assert metrics["num_queries"] == 2
        
        # Metrics should be between 0 and 1
        assert 0 <= metrics["precision@k"] <= 1
        assert 0 <= metrics["recall@k"] <= 1
        assert 0 <= metrics["f1@k"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])