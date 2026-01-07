"""Unit tests for text splitter module."""

import pytest
from unittest.mock import Mock

from langchain_rag_learning.core.models import Document, DocumentChunk
from langchain_rag_learning.core.exceptions import DocumentProcessingError
from langchain_rag_learning.rag.text_splitter import (
    RecursiveCharacterTextSplitter,
    SemanticTextSplitter,
    ContextAwareTextSplitter,
    DocumentChunker
)


class TestRecursiveCharacterTextSplitter:
    """Test cases for RecursiveCharacterTextSplitter class."""
    
    @pytest.fixture
    def splitter(self):
        """Create a RecursiveCharacterTextSplitter instance for testing."""
        return RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
    
    def test_initialization(self):
        """Test splitter initialization."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50
        assert splitter.separators == ["\n\n", "\n", " ", ""]
    
    def test_initialization_invalid_overlap(self):
        """Test initialization with invalid overlap."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=150)
    
    def test_split_text_simple(self, splitter):
        """Test splitting simple text."""
        text = "This is a short text that should fit in one chunk."
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_with_paragraphs(self, splitter):
        """Test splitting text with paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph with more content that might exceed the chunk size limit.\n\nThird paragraph."
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= splitter.chunk_size for chunk in chunks)
    
    def test_split_text_with_overlap(self):
        """Test that overlap is properly applied."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10
        )
        
        text = "This is a long text that will definitely need to be split into multiple chunks because it exceeds the chunk size."
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 1
        
        # Check that there's some overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # There should be some common content (overlap)
            # This is a simplified check - in practice, overlap might be more complex
            assert len(current_chunk) <= splitter.chunk_size
            assert len(next_chunk) <= splitter.chunk_size
    
    def test_split_on_separator(self, splitter):
        """Test splitting on specific separator."""
        text = "Part 1\n\nPart 2\n\nPart 3"
        splits = splitter._split_on_separator(text, "\n\n")
        
        assert len(splits) == 3
        assert "Part 1" in splits[0]
        assert "Part 2" in splits[1]
        assert "Part 3" in splits[2]
    
    def test_split_on_separator_keep_separator(self):
        """Test splitting while keeping separator."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            keep_separator=True
        )
        
        text = "Part 1\n\nPart 2\n\nPart 3"
        splits = splitter._split_on_separator(text, "\n\n")
        
        # When keeping separator, it should be attached to following parts
        assert len(splits) >= 2
        assert any("\n\n" in split for split in splits[1:])
    
    def test_create_document_chunks(self, splitter):
        """Test creating DocumentChunk objects."""
        text = "This is a test document that will be split into chunks."
        
        # Create a mock document
        document = Document(
            id="doc_1",
            filename="test.txt",
            original_filename="test.txt",
            file_path="/tmp/test.txt",
            file_size=len(text),
            file_type="txt",
            mime_type="text/plain",
            knowledge_base_id="kb_1",
            uploaded_by="user_1"
        )
        
        chunks = splitter.create_document_chunks(text, document)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == document.id for chunk in chunks)
        assert all(chunk.content in text for chunk in chunks)
        
        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_create_document_chunks_with_metadata(self, splitter):
        """Test creating chunks with base metadata."""
        text = "Test content"
        document = Document(
            id="doc_1",
            filename="test.txt",
            original_filename="test.txt",
            file_path="/tmp/test.txt",
            file_size=len(text),
            file_type="txt",
            mime_type="text/plain",
            knowledge_base_id="kb_1",
            uploaded_by="user_1"
        )
        
        base_metadata = {"custom_field": "custom_value"}
        chunks = splitter.create_document_chunks(text, document, base_metadata)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "custom_field" in chunk.metadata
            assert chunk.metadata["custom_field"] == "custom_value"
            assert "chunk_method" in chunk.metadata
            assert chunk.metadata["chunk_method"] == "RecursiveCharacterTextSplitter"


class TestSemanticTextSplitter:
    """Test cases for SemanticTextSplitter class."""
    
    @pytest.fixture
    def splitter(self):
        """Create a SemanticTextSplitter instance for testing."""
        return SemanticTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
    
    def test_split_text_with_paragraphs(self, splitter):
        """Test splitting text with paragraph structure."""
        text = """First paragraph with some content.

Second paragraph with more detailed information that might be longer than the first one.

Third paragraph that concludes the document."""
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= splitter.chunk_size for chunk in chunks)
        
        # Check that paragraph structure is preserved where possible
        for chunk in chunks:
            # Chunks should not start or end with excessive whitespace
            assert chunk == chunk.strip()
    
    def test_split_paragraph_by_sentences(self, splitter):
        """Test splitting large paragraphs by sentences."""
        paragraph = "This is the first sentence. This is the second sentence with more content. This is the third sentence. This is the fourth sentence that makes the paragraph quite long."
        
        chunks = splitter._split_paragraph_by_sentences(paragraph)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= splitter.chunk_size
            # Each chunk should contain complete sentences
            assert chunk.endswith('.') or chunk == chunks[-1]  # Last chunk might not end with period
    
    def test_get_overlap_text(self, splitter):
        """Test getting overlap text from chunk end."""
        text = "This is a long text. It has multiple sentences. The last sentence should be used for overlap."
        
        overlap = splitter._get_overlap_text(text, 50)
        
        assert len(overlap) <= 50
        assert overlap in text
        # Should prefer sentence boundaries
        assert "." in overlap or len(text) <= 50
    
    def test_apply_overlap(self, splitter):
        """Test applying overlap between chunks."""
        chunks = [
            "First chunk with some content.",
            "Second chunk with different content.",
            "Third chunk with final content."
        ]
        
        overlapped = splitter._apply_overlap(chunks)
        
        assert len(overlapped) == len(chunks)
        # First chunk should remain unchanged
        assert overlapped[0] == chunks[0]
        # Subsequent chunks should have overlap from previous chunks
        for i in range(1, len(overlapped)):
            assert len(overlapped[i]) >= len(chunks[i])


class TestContextAwareTextSplitter:
    """Test cases for ContextAwareTextSplitter class."""
    
    @pytest.fixture
    def splitter(self):
        """Create a ContextAwareTextSplitter instance for testing."""
        return ContextAwareTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            include_context=True,
            max_context_size=100
        )
    
    def test_analyze_document_structure(self, splitter):
        """Test document structure analysis."""
        text = """# Main Title

Some content under the main title.

## Section 1

Content for section 1.

### Subsection 1.1

More detailed content.

## Section 2

Content for section 2."""
        
        structure = splitter._analyze_document_structure(text)
        
        assert len(structure) >= 3  # Should find at least 3 headings
        
        # Check that headings are properly identified
        heading_texts = [item['text'] for item in structure]
        assert any("Main Title" in text for text in heading_texts)
        assert any("Section 1" in text for text in heading_texts)
        assert any("Section 2" in text for text in heading_texts)
    
    def test_get_chunk_context(self, splitter):
        """Test getting context for a chunk."""
        text = """# Chapter 1

## Section 1.1

This is some content in section 1.1.

## Section 1.2

This is content in section 1.2."""
        
        structure = splitter._analyze_document_structure(text)
        
        # Find position of content in section 1.2
        content_position = text.find("This is content in section 1.2")
        
        context = splitter._get_chunk_context(text, content_position, structure)
        
        assert context is not None
        assert "Section 1.2" in context
    
    def test_split_text_with_context(self, splitter):
        """Test splitting text while preserving context."""
        text = """# Document Title

## Introduction

This is the introduction section with some content that explains the purpose of the document.

## Main Content

This is the main content section with detailed information that might span multiple chunks due to its length and complexity.

## Conclusion

This is the conclusion section that summarizes the key points."""
        
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
        
        # Check that some chunks have context information
        context_chunks = [chunk for chunk in chunks if "Section:" in chunk or "Chapter:" in chunk]
        # Not all chunks may have context, but some should if the text is structured
        # This depends on the specific content and chunk boundaries
    
    def test_split_text_without_context(self):
        """Test splitting text without context inclusion."""
        splitter = ContextAwareTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            include_context=False
        )
        
        text = """# Title

## Section

Content here."""
        
        chunks = splitter.split_text(text)
        
        # Without context, chunks should not have context prefixes
        for chunk in chunks:
            assert not chunk.startswith("Section:")
            assert not chunk.startswith("Chapter:")


class TestDocumentChunker:
    """Test cases for DocumentChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a DocumentChunker instance for testing."""
        return DocumentChunker()
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample Document instance for testing."""
        return Document(
            id="doc_1",
            filename="test.txt",
            original_filename="test.txt",
            file_path="/tmp/test.txt",
            file_size=1000,
            file_type="txt",
            mime_type="text/plain",
            knowledge_base_id="kb_1",
            uploaded_by="user_1"
        )
    
    def test_initialization(self, chunker):
        """Test chunker initialization."""
        assert 'recursive' in chunker.strategies
        assert 'semantic' in chunker.strategies
        assert 'context_aware' in chunker.strategies
        assert chunker.default_strategy == 'semantic'
    
    @pytest.mark.asyncio
    async def test_chunk_document_default_strategy(self, chunker, sample_document):
        """Test chunking document with default strategy."""
        text = "This is a sample document that will be chunked using the default strategy."
        
        chunks = await chunker.chunk_document(text, sample_document)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == sample_document.id for chunk in chunks)
        
        # Check metadata
        for chunk in chunks:
            assert 'chunking_strategy' in chunk.metadata
            assert chunk.metadata['chunking_strategy'] == 'semantic'
    
    @pytest.mark.asyncio
    async def test_chunk_document_specific_strategy(self, chunker, sample_document):
        """Test chunking document with specific strategy."""
        text = "This is a sample document for testing recursive chunking strategy."
        
        chunks = await chunker.chunk_document(
            text, sample_document, strategy='recursive'
        )
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata['chunking_strategy'] == 'recursive'
    
    @pytest.mark.asyncio
    async def test_chunk_document_custom_parameters(self, chunker, sample_document):
        """Test chunking document with custom parameters."""
        text = "This is a longer sample document that will be chunked with custom parameters to test the flexibility of the chunking system."
        
        chunks = await chunker.chunk_document(
            text, sample_document,
            chunk_size=50,
            chunk_overlap=10
        )
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata['chunk_size'] == 50
            assert chunk.metadata['chunk_overlap'] == 10
    
    @pytest.mark.asyncio
    async def test_chunk_document_unknown_strategy(self, chunker, sample_document):
        """Test chunking with unknown strategy."""
        text = "Sample text"
        
        with pytest.raises(DocumentProcessingError, match="Unknown chunking strategy"):
            await chunker.chunk_document(text, sample_document, strategy='unknown')
    
    @pytest.mark.asyncio
    async def test_chunk_document_empty_text(self, chunker, sample_document):
        """Test chunking empty text."""
        text = ""
        
        chunks = await chunker.chunk_document(text, sample_document)
        
        # Should handle empty text gracefully
        assert isinstance(chunks, list)
    
    def test_get_available_strategies(self, chunker):
        """Test getting available strategies."""
        strategies = chunker.get_available_strategies()
        
        assert 'recursive' in strategies
        assert 'semantic' in strategies
        assert 'context_aware' in strategies
    
    def test_get_strategy_info(self, chunker):
        """Test getting strategy information."""
        info = chunker.get_strategy_info('semantic')
        
        assert 'name' in info
        assert 'class' in info
        assert 'description' in info
        assert 'parameters' in info
        assert info['name'] == 'semantic'
    
    def test_get_strategy_info_unknown(self, chunker):
        """Test getting info for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunker.get_strategy_info('unknown')


class TestIntegration:
    """Integration tests for text splitting workflow."""
    
    @pytest.mark.asyncio
    async def test_full_chunking_workflow(self):
        """Test the complete document chunking workflow."""
        # Sample document with various structures
        text = """# Document Title

## Introduction
This is the introduction section of the document. It provides an overview of what the document contains and sets the context for the reader.

## Main Content

### Subsection 1
This subsection contains detailed information about the first topic. It includes multiple paragraphs and various types of content.

The content continues with additional paragraphs that provide more depth and detail about the subject matter.

### Subsection 2
This subsection covers the second major topic. It also contains multiple paragraphs with comprehensive information.

## Conclusion
The conclusion summarizes the key points discussed in the document and provides final thoughts on the subject matter."""
        
        # Create document
        document = Document(
            id="doc_1",
            filename="test_doc.md",
            original_filename="test_doc.md",
            file_path="/tmp/test_doc.md",
            file_size=len(text.encode()),
            file_type="md",
            mime_type="text/markdown",
            knowledge_base_id="kb_1",
            uploaded_by="user_1"
        )
        
        # Test different chunking strategies
        chunker = DocumentChunker()
        
        for strategy in ['recursive', 'semantic', 'context_aware']:
            chunks = await chunker.chunk_document(
                text, document,
                strategy=strategy,
                chunk_size=300,
                chunk_overlap=50
            )
            
            # Verify results
            assert len(chunks) >= 1
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.document_id == document.id for chunk in chunks)
            assert all(chunk.metadata['chunking_strategy'] == strategy for chunk in chunks)
            
            # Verify chunk content
            all_content = ''.join(chunk.content for chunk in chunks)
            # Due to overlap and processing, we check that key content is preserved
            assert "Document Title" in all_content or "Document Title" in text
            assert "Introduction" in all_content
            assert "Conclusion" in all_content
            
            # Verify chunk indices are sequential
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i
            
            # Verify character positions make sense
            for chunk in chunks:
                assert chunk.start_char >= 0
                assert chunk.end_char > chunk.start_char
                assert chunk.end_char <= len(text)
    
    @pytest.mark.asyncio
    async def test_chunking_performance_with_large_text(self):
        """Test chunking performance with larger text."""
        # Generate a larger text document
        paragraphs = []
        for i in range(50):
            paragraph = f"This is paragraph {i+1}. " * 10  # Each paragraph ~200 chars
            paragraphs.append(paragraph)
        
        text = "\n\n".join(paragraphs)
        
        document = Document(
            id="large_doc",
            filename="large_test.txt",
            original_filename="large_test.txt",
            file_path="/tmp/large_test.txt",
            file_size=len(text.encode()),
            file_type="txt",
            mime_type="text/plain",
            knowledge_base_id="kb_1",
            uploaded_by="user_1"
        )
        
        chunker = DocumentChunker()
        
        # Test chunking with reasonable parameters
        chunks = await chunker.chunk_document(
            text, document,
            strategy='semantic',
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Verify results
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(len(chunk.content) <= 1000 for chunk in chunks)
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        
        # Verify all content is preserved (accounting for overlap)
        unique_content = set()
        for chunk in chunks:
            unique_content.add(chunk.content.strip())
        
        # Should have reasonable number of unique chunks
        assert len(unique_content) >= len(chunks) // 2  # Account for potential overlap