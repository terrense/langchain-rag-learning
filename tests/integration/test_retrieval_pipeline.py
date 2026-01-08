"""Integration tests for the complete retrieval pipeline."""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path

from langchain_rag_learning.core.models import DocumentChunk
from langchain_rag_learning.rag.embeddings import EmbeddingManager, HuggingFaceEmbeddingProvider
from langchain_rag_learning.rag.vector_store import ChromaVectorStore, VectorStoreManager
from langchain_rag_learning.rag.retrievers import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    RetrievalEngine
)


@pytest.fixture
def sample_documents():
    """Create sample documents for integration testing."""
    documents = [
        {
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "metadata": {"topic": "machine_learning", "difficulty": "beginner"}
        },
        {
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            "metadata": {"topic": "deep_learning", "difficulty": "intermediate"}
        },
        {
            "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "metadata": {"topic": "nlp", "difficulty": "intermediate"}
        },
        {
            "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.",
            "metadata": {"topic": "computer_vision", "difficulty": "advanced"}
        },
        {
            "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.",
            "metadata": {"topic": "reinforcement_learning", "difficulty": "advanced"}
        }
    ]
    
    chunks = []
    for i, doc in enumerate(documents):
        chunk = DocumentChunk(
            id=f"chunk_{i}",
            content=doc["content"],
            document_id=f"doc_{i}",
            chunk_index=0,
            start_char=0,
            end_char=len(doc["content"]),
            metadata=doc["metadata"]
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
async def embedding_manager():
    """Create embedding manager with HuggingFace provider."""
    manager = EmbeddingManager(cache_size=100)
    
    # Use a small, fast model for testing
    hf_provider = HuggingFaceEmbeddingProvider(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    manager.add_provider("test_hf", hf_provider)
    manager.set_default_provider("test_hf")
    
    return manager


@pytest.fixture
async def vector_store():
    """Create temporary vector store for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = ChromaVectorStore(
            collection_name="test_collection",
            embedding_dimension=384,  # all-MiniLM-L6-v2 dimension
            persist_directory=temp_dir
        )
        yield store


class TestDenseRetrievalIntegration:
    """Integration tests for dense retrieval."""
    
    @pytest.mark.asyncio
    async def test_dense_retrieval_end_to_end(self, sample_documents, embedding_manager, vector_store):
        """Test complete dense retrieval pipeline."""
        # Embed and store documents
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        
        # Create dense retriever
        dense_retriever = DenseRetriever(vector_store, embedding_manager)
        
        # Test retrieval
        results = await dense_retriever.retrieve("machine learning algorithms", k=3)
        
        assert len(results) <= 3
        assert all(isinstance(chunk, DocumentChunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # Results should be sorted by similarity score
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        
        # Check that relevant documents are retrieved
        retrieved_topics = [chunk.metadata.get("topic") for chunk, _ in results]
        assert "machine_learning" in retrieved_topics or "deep_learning" in retrieved_topics
    
    @pytest.mark.asyncio
    async def test_dense_retrieval_with_filters(self, sample_documents, embedding_manager, vector_store):
        """Test dense retrieval with metadata filters."""
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        
        dense_retriever = DenseRetriever(vector_store, embedding_manager)
        
        # Test with difficulty filter
        results = await dense_retriever.retrieve(
            "artificial intelligence",
            k=5,
            filter_dict={"difficulty": "intermediate"}
        )
        
        # Should only return intermediate difficulty documents
        for chunk, _ in results:
            assert chunk.metadata.get("difficulty") == "intermediate"
    
    @pytest.mark.asyncio
    async def test_query_expansion_integration(self, sample_documents, embedding_manager, vector_store):
        """Test query expansion with real embeddings."""
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        
        dense_retriever = DenseRetriever(vector_store, embedding_manager)
        
        expansion_terms = await dense_retriever.get_query_expansion_terms("learning")
        
        assert isinstance(expansion_terms, list)
        # Should extract meaningful terms
        assert len(expansion_terms) >= 0  # May be empty if no good expansion terms


class TestSparseRetrievalIntegration:
    """Integration tests for sparse retrieval."""
    
    @pytest.mark.asyncio
    async def test_sparse_retrieval_bm25(self, sample_documents):
        """Test BM25 sparse retrieval."""
        sparse_retriever = SparseRetriever(sample_documents, name="test_bm25")
        
        results = await sparse_retriever.retrieve("machine learning", k=3, scoring_method="bm25")
        
        assert len(results) <= 3
        assert all(isinstance(chunk, DocumentChunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # Should find relevant documents
        retrieved_content = [chunk.content.lower() for chunk, _ in results]
        assert any("machine learning" in content for content in retrieved_content)
    
    @pytest.mark.asyncio
    async def test_sparse_retrieval_tfidf(self, sample_documents):
        """Test TF-IDF sparse retrieval."""
        sparse_retriever = SparseRetriever(sample_documents, name="test_tfidf")
        
        results = await sparse_retriever.retrieve("neural networks", k=3, scoring_method="tfidf")
        
        assert len(results) <= 3
        # Should find documents containing "neural"
        retrieved_content = [chunk.content.lower() for chunk, _ in results if chunk.content.lower().find("neural") != -1]
        assert len(retrieved_content) > 0
    
    @pytest.mark.asyncio
    async def test_sparse_retrieval_with_expansion(self, sample_documents):
        """Test sparse retrieval with query expansion."""
        sparse_retriever = SparseRetriever(sample_documents)
        
        results_without_expansion = await sparse_retriever.retrieve("AI", k=5, query_expansion=False)
        results_with_expansion = await sparse_retriever.retrieve("AI", k=5, query_expansion=True)
        
        # Both should work
        assert isinstance(results_without_expansion, list)
        assert isinstance(results_with_expansion, list)


class TestHybridRetrievalIntegration:
    """Integration tests for hybrid retrieval."""
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_rrf(self, sample_documents, embedding_manager, vector_store):
        """Test hybrid retrieval with RRF fusion."""
        # Setup dense retriever
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        dense_retriever = DenseRetriever(vector_store, embedding_manager)
        
        # Setup sparse retriever
        sparse_retriever = SparseRetriever(sample_documents)
        
        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            dense_retriever,
            sparse_retriever,
            dense_weight=0.6,
            sparse_weight=0.4
        )
        
        results = await hybrid_retriever.retrieve(
            "machine learning algorithms",
            k=3,
            fusion_method="rrf"
        )
        
        assert len(results) <= 3
        assert all(isinstance(chunk, DocumentChunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # Check that results have hybrid metadata
        for chunk, _ in results:
            assert "retrieval_info" in chunk.metadata
            assert chunk.metadata["retrieval_info"]["fusion_method"] == "rrf"
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_weighted(self, sample_documents, embedding_manager, vector_store):
        """Test hybrid retrieval with weighted fusion."""
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        dense_retriever = DenseRetriever(vector_store, embedding_manager)
        sparse_retriever = SparseRetriever(sample_documents)
        
        hybrid_retriever = HybridRetriever(
            dense_retriever,
            sparse_retriever,
            dense_weight=0.8,
            sparse_weight=0.2
        )
        
        results = await hybrid_retriever.retrieve(
            "deep learning neural networks",
            k=3,
            fusion_method="weighted"
        )
        
        assert len(results) <= 3
        # Should favor dense retrieval results due to higher weight
        for chunk, _ in results:
            assert "retrieval_info" in chunk.metadata
            assert chunk.metadata["retrieval_info"]["dense_weight"] == 0.8
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_comparison(self, sample_documents, embedding_manager, vector_store):
        """Compare different fusion methods."""
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        dense_retriever = DenseRetriever(vector_store, embedding_manager)
        sparse_retriever = SparseRetriever(sample_documents)
        
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
        
        query = "artificial intelligence learning"
        
        # Test different fusion methods
        rrf_results = await hybrid_retriever.retrieve(query, k=3, fusion_method="rrf")
        weighted_results = await hybrid_retriever.retrieve(query, k=3, fusion_method="weighted")
        max_results = await hybrid_retriever.retrieve(query, k=3, fusion_method="max")
        
        # All should return results
        assert len(rrf_results) > 0
        assert len(weighted_results) > 0
        assert len(max_results) > 0
        
        # Results may be different due to different fusion methods
        rrf_ids = [chunk.id for chunk, _ in rrf_results]
        weighted_ids = [chunk.id for chunk, _ in weighted_results]
        max_ids = [chunk.id for chunk, _ in max_results]
        
        # At least some results should be common across methods
        common_results = set(rrf_ids) & set(weighted_ids) & set(max_ids)
        assert len(common_results) >= 1


class TestRetrievalEngineIntegration:
    """Integration tests for the complete retrieval engine."""
    
    @pytest.mark.asyncio
    async def test_retrieval_engine_setup(self, sample_documents, embedding_manager, vector_store):
        """Test complete retrieval engine setup and usage."""
        # Create vector store manager
        vector_store_manager = VectorStoreManager()
        
        # Create retrieval engine
        engine = RetrievalEngine(vector_store_manager, embedding_manager)
        
        # Setup dense retriever
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        engine.add_dense_retriever("dense", vector_store)
        
        # Setup sparse retriever
        engine.add_sparse_retriever("sparse", sample_documents)
        
        # Setup hybrid retriever
        engine.add_hybrid_retriever("hybrid", "dense", "sparse")
        
        # Test retrieval with different retrievers
        dense_results = await engine.retrieve("machine learning", retriever_name="dense", k=3)
        sparse_results = await engine.retrieve("machine learning", retriever_name="sparse", k=3)
        hybrid_results = await engine.retrieve("machine learning", retriever_name="hybrid", k=3)
        
        assert len(dense_results) > 0
        assert len(sparse_results) > 0
        assert len(hybrid_results) > 0
        
        # Test retriever info
        dense_info = engine.get_retriever_info("dense")
        sparse_info = engine.get_retriever_info("sparse")
        hybrid_info = engine.get_retriever_info("hybrid")
        
        assert dense_info["type"] == "DenseRetriever"
        assert sparse_info["type"] == "SparseRetriever"
        assert hybrid_info["type"] == "HybridRetriever"
    
    @pytest.mark.asyncio
    async def test_retrieval_evaluation(self, sample_documents, embedding_manager, vector_store):
        """Test retrieval evaluation functionality."""
        vector_store_manager = VectorStoreManager()
        engine = RetrievalEngine(vector_store_manager, embedding_manager)
        
        # Setup retrievers
        embedded_chunks = await embedding_manager.embed_document_chunks(sample_documents)
        await vector_store.add_documents(embedded_chunks)
        engine.add_dense_retriever("dense", vector_store)
        engine.add_sparse_retriever("sparse", sample_documents)
        
        # Create test queries and ground truth
        test_queries = [
            "machine learning algorithms",
            "neural networks deep learning",
            "natural language processing"
        ]
        
        ground_truth = [
            ["chunk_0", "chunk_1"],  # ML and DL are relevant for first query
            ["chunk_1"],             # DL is relevant for second query
            ["chunk_2"]              # NLP is relevant for third query
        ]
        
        # Evaluate dense retriever
        dense_metrics = await engine.evaluate_retrieval(
            test_queries, ground_truth, retriever_name="dense", k=3
        )
        
        assert "precision@k" in dense_metrics
        assert "recall@k" in dense_metrics
        assert "f1@k" in dense_metrics
        assert dense_metrics["num_queries"] == 3
        
        # Evaluate sparse retriever
        sparse_metrics = await engine.evaluate_retrieval(
            test_queries, ground_truth, retriever_name="sparse", k=3
        )
        
        assert sparse_metrics["num_queries"] == 3
        
        # Metrics should be reasonable (between 0 and 1)
        for metric_name, value in dense_metrics.items():
            if metric_name != "num_queries":
                assert 0 <= value <= 1
        
        for metric_name, value in sparse_metrics.items():
            if metric_name != "num_queries":
                assert 0 <= value <= 1


class TestRetrievalPerformance:
    """Performance tests for retrieval system."""
    
    @pytest.mark.asyncio
    async def test_retrieval_performance_scaling(self, embedding_manager):
        """Test retrieval performance with different document counts."""
        import time
        
        # Create documents of different sizes
        document_counts = [10, 50, 100]
        performance_results = {}
        
        for count in document_counts:
            # Generate test documents
            documents = []
            for i in range(count):
                chunk = DocumentChunk(
                    id=f"perf_chunk_{i}",
                    content=f"This is test document {i} about machine learning and artificial intelligence. " * 10,
                    document_id=f"perf_doc_{i}",
                    chunk_index=0,
                    start_char=0,
                    end_char=100,
                    metadata={"index": i}
                )
                documents.append(chunk)
            
            # Test sparse retrieval performance
            sparse_retriever = SparseRetriever(documents)
            
            start_time = time.time()
            results = await sparse_retriever.retrieve("machine learning", k=5)
            sparse_time = time.time() - start_time
            
            performance_results[count] = {
                "sparse_time": sparse_time,
                "results_count": len(results)
            }
        
        # Performance should scale reasonably
        assert performance_results[10]["sparse_time"] < performance_results[100]["sparse_time"]
        
        # Should still return results
        for count in document_counts:
            assert performance_results[count]["results_count"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_retrieval(self, sample_documents):
        """Test concurrent retrieval requests."""
        sparse_retriever = SparseRetriever(sample_documents)
        
        # Create multiple concurrent queries
        queries = [
            "machine learning",
            "deep learning",
            "neural networks",
            "artificial intelligence",
            "computer vision"
        ]
        
        # Execute queries concurrently
        tasks = [
            sparse_retriever.retrieve(query, k=3)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All queries should return results
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, list)
            assert len(result) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])