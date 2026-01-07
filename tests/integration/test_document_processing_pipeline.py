"""Integration tests for the complete document processing pipeline."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from langchain_rag_learning.core.models import Document, DocumentStatus, DocumentChunk
from langchain_rag_learning.rag.document_processor import DocumentProcessor
from langchain_rag_learning.rag.text_splitter import DocumentChunker
from langchain_rag_learning.rag.embeddings import EmbeddingManager
from langchain_rag_learning.rag.vector_store import VectorStoreManager


class TestDocumentProcessingPipeline:
    """Integration tests for the complete document processing pipeline."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        documents = {}
        
        # Text document
        text_content = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.

Key Concepts:
- Supervised Learning
- Unsupervised Learning  
- Reinforcement Learning

Applications include natural language processing, computer vision, and recommendation systems."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text_content)
            documents['text'] = {
                'path': f.name,
                'content': text_content,
                'document': Document(
                    filename="ml_intro.txt",
                    original_filename="ml_intro.txt",
                    file_path=f.name,
                    file_size=len(text_content.encode()),
                    file_type="txt",
                    mime_type="text/plain",
                    knowledge_base_id="kb_1",
                    uploaded_by="user_1"
                )
            }
        
        # Markdown document
        markdown_content = """# Deep Learning Guide

## Neural Networks
Neural networks are computing systems inspired by biological neural networks.

### Types of Neural Networks
1. **Feedforward Networks**
   - Simple architecture
   - Information flows in one direction
   
2. **Convolutional Networks (CNNs)**
   - Excellent for image processing
   - Use convolution operations
   
3. **Recurrent Networks (RNNs)**
   - Handle sequential data
   - Have memory capabilities

## Training Process
The training process involves:
- Forward propagation
- Loss calculation
- Backpropagation
- Parameter updates

## Conclusion
Deep learning has revolutionized many fields including computer vision, natural language processing, and speech recognition."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            documents['markdown'] = {
                'path': f.name,
                'content': markdown_content,
                'document': Document(
                    filename="deep_learning.md",
                    original_filename="deep_learning.md",
                    file_path=f.name,
                    file_size=len(markdown_content.encode()),
                    file_type="md",
                    mime_type="text/markdown",
                    knowledge_base_id="kb_1",
                    uploaded_by="user_1"
                )
            }
        
        yield documents
        
        # Cleanup
        for doc_info in documents.values():
            try:
                os.unlink(doc_info['path'])
            except FileNotFoundError:
                pass
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider for testing."""
        provider = AsyncMock()
        provider.model_name = "test_model"
        provider.get_embedding_dimension.return_value = 384
        
        # Generate deterministic embeddings based on text length
        def mock_embed_documents(texts):
            embeddings = []
            for text in texts:
                # Create a simple embedding based on text characteristics
                embedding = [0.1] * 384
                embedding[0] = len(text) / 1000.0  # Vary first dimension by length
                embedding[1] = text.count(' ') / 100.0  # Vary second dimension by word count
                embeddings.append(embedding)
            return embeddings
        
        def mock_embed_query(text):
            embedding = [0.1] * 384
            embedding[0] = len(text) / 1000.0
            embedding[1] = text.count(' ') / 100.0
            return embedding
        
        provider.embed_documents.side_effect = mock_embed_documents
        provider.embed_query.side_effect = mock_embed_query
        
        return provider
    
    @pytest.mark.asyncio
    async def test_complete_document_processing_pipeline(self, sample_documents, mock_embedding_provider):
        """Test the complete document processing pipeline from raw files to searchable vectors."""
        
        # Initialize components
        document_processor = DocumentProcessor()
        chunker = DocumentChunker()
        
        # Create embedding manager with mock provider
        embedding_manager = EmbeddingManager(cache_size=100)
        embedding_manager.add_provider("mock", mock_embedding_provider)
        embedding_manager.set_default_provider("mock")
        
        # Process each document through the complete pipeline
        all_chunks = []
        
        for doc_type, doc_info in sample_documents.items():
            document = doc_info['document']
            file_path = doc_info['path']
            
            # Step 1: Extract and clean text
            extracted_text, metadata = await document_processor.process_document(file_path, document)
            
            # Verify text extraction
            assert extracted_text is not None
            assert len(extracted_text) > 0
            assert isinstance(metadata, dict)
            
            # Verify content is preserved
            if doc_type == 'text':
                assert "Machine learning" in extracted_text
                assert "supervised learning" in extracted_text.lower()
            elif doc_type == 'markdown':
                assert "Deep Learning" in extracted_text
                assert "Neural Networks" in extracted_text
            
            # Step 2: Chunk the document
            chunks = await chunker.chunk_document(
                extracted_text,
                document,
                strategy='semantic',
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Verify chunking
            assert len(chunks) >= 1
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.document_id == document.id for chunk in chunks)
            
            # Verify chunk content
            combined_content = ' '.join(chunk.content for chunk in chunks)
            if doc_type == 'text':
                assert "Machine learning" in combined_content
            elif doc_type == 'markdown':
                assert "Deep Learning" in combined_content
            
            # Step 3: Generate embeddings
            embedded_chunks = await embedding_manager.embed_document_chunks(
                chunks,
                provider="mock",
                use_cache=True
            )
            
            # Verify embeddings
            assert len(embedded_chunks) == len(chunks)
            for chunk in embedded_chunks:
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 384
                assert 'embedding_info' in chunk.metadata
                assert chunk.metadata['embedding_info']['provider'] == 'mock'
            
            all_chunks.extend(embedded_chunks)
        
        # Verify we have chunks from both documents
        assert len(all_chunks) >= 2
        
        # Verify chunks have different embeddings (based on content)
        embeddings = [chunk.embedding for chunk in all_chunks]
        assert len(set(tuple(emb) for emb in embeddings)) > 1  # Should have different embeddings
        
        # Test that embeddings reflect content differences
        text_chunks = [chunk for chunk in all_chunks if chunk.document_id == sample_documents['text']['document'].id]
        md_chunks = [chunk for chunk in all_chunks if chunk.document_id == sample_documents['markdown']['document'].id]
        
        assert len(text_chunks) > 0
        assert len(md_chunks) > 0
        
        # Embeddings should be different between documents
        text_embedding = text_chunks[0].embedding
        md_embedding = md_chunks[0].embedding
        assert text_embedding != md_embedding
    
    @pytest.mark.asyncio
    async def test_pipeline_with_vector_storage(self, sample_documents, mock_embedding_provider):
        """Test the pipeline including vector storage and retrieval."""
        
        # Initialize all components
        document_processor = DocumentProcessor()
        chunker = DocumentChunker()
        
        embedding_manager = EmbeddingManager(cache_size=100)
        embedding_manager.add_provider("mock", mock_embedding_provider)
        embedding_manager.set_default_provider("mock")
        
        # Use in-memory vector store for testing
        vector_store_manager = VectorStoreManager()
        
        # Process documents and store in vector database
        all_chunks = []
        
        for doc_info in sample_documents.values():
            document = doc_info['document']
            file_path = doc_info['path']
            
            # Process document
            extracted_text, metadata = await document_processor.process_document(file_path, document)
            
            # Chunk document
            chunks = await chunker.chunk_document(
                extracted_text,
                document,
                strategy='semantic',
                chunk_size=300,
                chunk_overlap=30
            )
            
            # Generate embeddings
            embedded_chunks = await embedding_manager.embed_document_chunks(chunks, provider="mock")
            
            all_chunks.extend(embedded_chunks)
        
        # Create vector store and add documents
        vector_store = vector_store_manager.create_store(
            store_type="chroma",
            collection_name="test_collection",
            embedding_dimension=384
        )
        
        # Add chunks to vector store
        chunk_ids = await vector_store.add_documents(all_chunks)
        assert len(chunk_ids) == len(all_chunks)
        
        # Test similarity search
        query_text = "What is machine learning?"
        query_embedding = await embedding_manager.embed_query(query_text, provider="mock")
        
        search_results = await vector_store.similarity_search(
            query_embedding,
            k=3
        )
        
        # Verify search results
        assert len(search_results) <= 3
        assert len(search_results) > 0
        
        for chunk, score in search_results:
            assert isinstance(chunk, DocumentChunk)
            assert isinstance(score, float)
            assert chunk.embedding is not None
            assert chunk.content is not None
        
        # Verify that relevant content is returned
        result_contents = [chunk.content.lower() for chunk, _ in search_results]
        combined_results = ' '.join(result_contents)
        
        # Should find content related to machine learning
        assert any(
            term in combined_results 
            for term in ['machine learning', 'artificial intelligence', 'algorithm']
        )
        
        # Test metadata filtering
        text_doc_id = sample_documents['text']['document'].id
        filtered_results = await vector_store.similarity_search(
            query_embedding,
            k=5,
            filter_dict={"document_id": text_doc_id}
        )
        
        # All results should be from the text document
        for chunk, _ in filtered_results:
            assert chunk.document_id == text_doc_id
        
        # Get collection stats
        stats = await vector_store.get_collection_stats()
        assert stats['document_count'] == len(all_chunks)
        assert stats['embedding_dimension'] == 384
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_embedding_provider):
        """Test error handling in the pipeline."""
        
        # Test with non-existent file
        document = Document(
            filename="nonexistent.txt",
            original_filename="nonexistent.txt",
            file_path="/nonexistent/path.txt",
            file_size=100,
            file_type="txt",
            mime_type="text/plain",
            knowledge_base_id="kb_1",
            uploaded_by="user_1"
        )
        
        document_processor = DocumentProcessor()
        
        # Should handle file not found gracefully
        with pytest.raises(Exception):  # Could be DocumentProcessingError or FileNotFoundError
            await document_processor.process_document("/nonexistent/path.txt", document)
    
    @pytest.mark.asyncio
    async def test_pipeline_with_different_chunking_strategies(self, sample_documents, mock_embedding_provider):
        """Test the pipeline with different chunking strategies."""
        
        document_processor = DocumentProcessor()
        chunker = DocumentChunker()
        
        embedding_manager = EmbeddingManager(cache_size=100)
        embedding_manager.add_provider("mock", mock_embedding_provider)
        embedding_manager.set_default_provider("mock")
        
        # Test with markdown document
        doc_info = sample_documents['markdown']
        document = doc_info['document']
        file_path = doc_info['path']
        
        # Extract text
        extracted_text, _ = await document_processor.process_document(file_path, document)
        
        # Test different chunking strategies
        strategies = ['recursive', 'semantic', 'context_aware']
        strategy_results = {}
        
        for strategy in strategies:
            chunks = await chunker.chunk_document(
                extracted_text,
                document,
                strategy=strategy,
                chunk_size=400,
                chunk_overlap=40
            )
            
            # Generate embeddings
            embedded_chunks = await embedding_manager.embed_document_chunks(chunks, provider="mock")
            
            strategy_results[strategy] = embedded_chunks
            
            # Verify basic properties
            assert len(embedded_chunks) >= 1
            assert all(chunk.embedding is not None for chunk in embedded_chunks)
            assert all(chunk.metadata['chunking_strategy'] == strategy for chunk in embedded_chunks)
        
        # Different strategies should potentially produce different numbers of chunks
        chunk_counts = {strategy: len(chunks) for strategy, chunks in strategy_results.items()}
        
        # All strategies should produce at least one chunk
        assert all(count >= 1 for count in chunk_counts.values())
        
        # Verify that chunks contain the expected content
        for strategy, chunks in strategy_results.items():
            combined_content = ' '.join(chunk.content for chunk in chunks)
            assert "Deep Learning" in combined_content
            assert "Neural Networks" in combined_content
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_with_large_document(self, mock_embedding_provider):
        """Test pipeline performance with a larger document."""
        
        # Create a larger document
        large_content = []
        for i in range(20):
            section = f"""
# Section {i+1}: Advanced Topic

This section covers advanced topic number {i+1} in great detail. 
It includes multiple paragraphs with comprehensive explanations.

## Subsection {i+1}.1: Technical Details

The technical details for this topic are quite extensive and require 
careful consideration of various factors and implementation strategies.

Key points include:
- Point 1: Important consideration for implementation
- Point 2: Performance optimization techniques  
- Point 3: Best practices and common pitfalls

## Subsection {i+1}.2: Practical Applications

Real-world applications of this topic demonstrate its importance in 
modern software development and system architecture.

Examples include distributed systems, microservices architecture, 
and cloud-native applications that leverage these concepts.

## Conclusion for Section {i+1}

This section has covered the fundamental and advanced aspects of 
topic {i+1}, providing both theoretical background and practical guidance.
"""
            large_content.append(section)
        
        full_content = '\n'.join(large_content)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(full_content)
            temp_path = f.name
        
        try:
            document = Document(
                filename="large_doc.md",
                original_filename="large_doc.md", 
                file_path=temp_path,
                file_size=len(full_content.encode()),
                file_type="md",
                mime_type="text/markdown",
                knowledge_base_id="kb_1",
                uploaded_by="user_1"
            )
            
            # Process through pipeline
            document_processor = DocumentProcessor()
            chunker = DocumentChunker()
            
            embedding_manager = EmbeddingManager(cache_size=1000)
            embedding_manager.add_provider("mock", mock_embedding_provider)
            embedding_manager.set_default_provider("mock")
            
            # Extract text
            extracted_text, metadata = await document_processor.process_document(temp_path, document)
            
            # Verify extraction
            assert len(extracted_text) > 10000  # Should be a substantial document
            assert "Section 1:" in extracted_text
            assert "Section 20:" in extracted_text
            
            # Chunk document
            chunks = await chunker.chunk_document(
                extracted_text,
                document,
                strategy='semantic',
                chunk_size=800,
                chunk_overlap=80
            )
            
            # Should create multiple chunks
            assert len(chunks) > 5
            assert len(chunks) < 100  # But not too many
            
            # Generate embeddings in batches
            embedded_chunks = await embedding_manager.embed_document_chunks(
                chunks,
                provider="mock",
                batch_size=10
            )
            
            # Verify all chunks have embeddings
            assert len(embedded_chunks) == len(chunks)
            assert all(chunk.embedding is not None for chunk in embedded_chunks)
            
            # Verify content distribution
            total_content_length = sum(len(chunk.content) for chunk in embedded_chunks)
            assert total_content_length > 0
            
            # Check that chunks have reasonable sizes
            chunk_sizes = [len(chunk.content) for chunk in embedded_chunks]
            assert max(chunk_sizes) <= 800 + 100  # Allow some flexibility
            assert min(chunk_sizes) > 50  # Shouldn't have tiny chunks
            
        finally:
            os.unlink(temp_path)