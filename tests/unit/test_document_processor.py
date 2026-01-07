"""Unit tests for document processor module."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from langchain_rag_learning.core.models import Document, DocumentStatus
from langchain_rag_learning.core.exceptions import DocumentProcessingError
from langchain_rag_learning.rag.document_processor import (
    DocumentProcessor,
    PDFParser,
    WordParser,
    TextParser,
    MarkdownParser,
    HTMLParser,
    MetadataExtractor,
    TextCleaner
)


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    @pytest.fixture
    def document_processor(self):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample Document instance for testing."""
        return Document(
            filename="test.txt",
            original_filename="test.txt",
            file_path="/tmp/test.txt",
            file_size=1024,
            file_type="txt",
            mime_type="text/plain",
            knowledge_base_id="kb_1",
            uploaded_by="user_1",
            status=DocumentStatus.PENDING
        )
    
    def test_get_file_type_from_declared(self, document_processor):
        """Test file type detection from declared type."""
        file_type = document_processor._get_file_type("/path/to/file.unknown", "pdf")
        assert file_type == "pdf"
    
    def test_get_file_type_from_extension(self, document_processor):
        """Test file type detection from file extension."""
        file_type = document_processor._get_file_type("/path/to/file.docx", "")
        assert file_type == "docx"
    
    def test_get_file_type_fallback(self, document_processor):
        """Test file type detection fallback to text."""
        file_type = document_processor._get_file_type("/path/to/file.unknown", "")
        assert file_type == "txt"
    
    @pytest.mark.asyncio
    async def test_process_document_success(self, document_processor, sample_document):
        """Test successful document processing."""
        with patch.object(document_processor.parsers['txt'], 'extract_text', new_callable=AsyncMock) as mock_extract:
            with patch.object(document_processor.text_cleaner, 'clean_text') as mock_clean:
                with patch.object(document_processor.metadata_extractor, 'extract_metadata', new_callable=AsyncMock) as mock_metadata:
                    
                    mock_extract.return_value = "Sample text content"
                    mock_clean.return_value = "Cleaned text content"
                    mock_metadata.return_value = {"word_count": 3}
                    
                    text, metadata = await document_processor.process_document("/tmp/test.txt", sample_document)
                    
                    assert text == "Cleaned text content"
                    assert metadata == {"word_count": 3}
                    mock_extract.assert_called_once_with("/tmp/test.txt")
                    mock_clean.assert_called_once_with("Sample text content")
    
    @pytest.mark.asyncio
    async def test_process_document_unsupported_type(self, document_processor, sample_document):
        """Test processing document with unsupported file type."""
        sample_document.file_type = "unsupported"
        
        with pytest.raises(DocumentProcessingError, match="No parser available"):
            await document_processor.process_document("/tmp/test.unsupported", sample_document)
    
    @pytest.mark.asyncio
    async def test_process_document_parser_error(self, document_processor, sample_document):
        """Test handling of parser errors."""
        with patch.object(document_processor.parsers['txt'], 'extract_text', new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = Exception("Parser error")
            
            with pytest.raises(DocumentProcessingError, match="Failed to process document"):
                await document_processor.process_document("/tmp/test.txt", sample_document)


class TestTextParser:
    """Test cases for TextParser class."""
    
    @pytest.fixture
    def text_parser(self):
        """Create a TextParser instance for testing."""
        return TextParser()
    
    @pytest.mark.asyncio
    async def test_extract_text_utf8(self, text_parser):
        """Test extracting text from UTF-8 file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write("Hello, world! 你好世界!")
            temp_path = f.name
        
        try:
            text = await text_parser.extract_text(temp_path)
            assert text == "Hello, world! 你好世界!"
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_extract_text_latin1(self, text_parser):
        """Test extracting text from Latin-1 encoded file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='latin-1', delete=False) as f:
            f.write("Café résumé")
            temp_path = f.name
        
        try:
            text = await text_parser.extract_text(temp_path)
            assert "Café" in text
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_extract_text_file_not_found(self, text_parser):
        """Test handling of non-existent file."""
        with pytest.raises(DocumentProcessingError):
            await text_parser.extract_text("/nonexistent/file.txt")


class TestMarkdownParser:
    """Test cases for MarkdownParser class."""
    
    @pytest.fixture
    def markdown_parser(self):
        """Create a MarkdownParser instance for testing."""
        return MarkdownParser()
    
    def test_process_markdown_headers(self, markdown_parser):
        """Test processing of markdown headers."""
        content = "# Main Title\n## Subtitle\n### Sub-subtitle"
        processed = markdown_parser._process_markdown(content)
        
        assert "= Main Title =" in processed
        assert "== Subtitle ==" in processed
        assert "=== Sub-subtitle ===" in processed
    
    def test_process_markdown_formatting(self, markdown_parser):
        """Test processing of markdown formatting."""
        content = "**bold** *italic* `code` [link](url)"
        processed = markdown_parser._process_markdown(content)
        
        assert "bold" in processed
        assert "italic" in processed
        assert "code" in processed
        assert "link" in processed
        assert "**" not in processed
        assert "*" not in processed.replace("•", "")  # Exclude bullet points
        assert "`" not in processed
        assert "[" not in processed
        assert "]" not in processed
        assert "(" not in processed
        assert ")" not in processed
    
    def test_process_markdown_lists(self, markdown_parser):
        """Test processing of markdown lists."""
        content = "- Item 1\n* Item 2\n+ Item 3\n1. Numbered item"
        processed = markdown_parser._process_markdown(content)
        
        # All list items should be converted to bullet points
        lines = processed.split('\n')
        list_lines = [line for line in lines if line.strip().startswith('•')]
        assert len(list_lines) == 4


class TestTextCleaner:
    """Test cases for TextCleaner class."""
    
    @pytest.fixture
    def text_cleaner(self):
        """Create a TextCleaner instance for testing."""
        return TextCleaner()
    
    def test_normalize_whitespace(self, text_cleaner):
        """Test whitespace normalization."""
        text = "Multiple   spaces\t\tand\ttabs"
        cleaned = text_cleaner._normalize_whitespace(text)
        assert "Multiple spaces and tabs" in cleaned
    
    def test_remove_unwanted_characters(self, text_cleaner):
        """Test removal of unwanted characters."""
        text = "Normal text\x00\x01\x02with control chars"
        cleaned = text_cleaner._remove_unwanted_characters(text)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "\x02" not in cleaned
        assert "Normal text" in cleaned
        assert "with control chars" in cleaned
    
    def test_fix_encoding_issues(self, text_cleaner):
        """Test fixing common encoding issues."""
        text = "Itâ€™s a â€œquoteâ€ with â€" dash"
        cleaned = text_cleaner._fix_encoding_issues(text)
        assert "It's" in cleaned
        assert '"quote"' in cleaned
        assert "—" in cleaned
    
    def test_normalize_line_breaks(self, text_cleaner):
        """Test line break normalization."""
        text = "Line 1\n\n\n\nLine 2\n   \n\nLine 3"
        cleaned = text_cleaner._normalize_line_breaks(text)
        
        # Should have at most double line breaks
        assert "\n\n\n" not in cleaned
        assert "Line 1\n\nLine 2\n\nLine 3" in cleaned
    
    def test_clean_text_complete(self, text_cleaner):
        """Test complete text cleaning process."""
        text = "  Multiple   spaces\t\tâ€™s text\n\n\n\nwith issues  "
        cleaned = text_cleaner.clean_text(text)
        
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace
        assert "Multiple spaces" in cleaned
        assert "'s text" in cleaned
        assert "\n\n\n" not in cleaned


class TestMetadataExtractor:
    """Test cases for MetadataExtractor class."""
    
    @pytest.fixture
    def metadata_extractor(self):
        """Create a MetadataExtractor instance for testing."""
        return MetadataExtractor()
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample Document instance for testing."""
        return Document(
            filename="test.txt",
            original_filename="test.txt",
            file_path="/tmp/test.txt",
            file_size=1024,
            file_type="txt",
            mime_type="text/plain",
            knowledge_base_id="kb_1",
            uploaded_by="user_1"
        )
    
    def test_analyze_text_content(self, metadata_extractor):
        """Test text content analysis."""
        text = "This is a test document.\nIt has multiple lines.\nAnd several words."
        metadata = metadata_extractor._analyze_text_content(text)
        
        assert metadata['character_count'] > 0
        assert metadata['word_count'] > 0
        assert metadata['line_count'] == 3
        assert metadata['paragraph_count'] == 3
        assert 'language_detected' in metadata
    
    def test_analyze_empty_text(self, metadata_extractor):
        """Test analysis of empty text."""
        metadata = metadata_extractor._analyze_text_content("")
        assert metadata == {}
    
    def test_detect_language_english(self, metadata_extractor):
        """Test English language detection."""
        text = "This is an English text sample with many words."
        language = metadata_extractor._detect_language(text)
        assert language == "en"
    
    def test_detect_language_chinese(self, metadata_extractor):
        """Test Chinese language detection."""
        text = "这是一个中文文本示例，包含很多中文字符。"
        language = metadata_extractor._detect_language(text)
        assert language == "zh"
    
    @pytest.mark.asyncio
    async def test_extract_metadata_complete(self, metadata_extractor, sample_document):
        """Test complete metadata extraction."""
        text_content = "Sample document content with multiple words and lines.\nSecond line here."
        
        metadata = await metadata_extractor.extract_metadata(
            "/tmp/test.txt", sample_document, text_content
        )
        
        # Check basic file metadata
        assert metadata['file_size_bytes'] == 1024
        assert metadata['file_extension'] == '.txt'
        assert metadata['mime_type'] == 'text/plain'
        assert metadata['original_filename'] == 'test.txt'
        
        # Check text analysis metadata
        assert 'character_count' in metadata
        assert 'word_count' in metadata
        assert 'line_count' in metadata
        assert 'language_detected' in metadata


@pytest.mark.skipif(
    not pytest.importorskip("PyPDF2", reason="PyPDF2 not available"),
    reason="PyPDF2 not installed"
)
class TestPDFParser:
    """Test cases for PDFParser class (requires PyPDF2)."""
    
    @pytest.fixture
    def pdf_parser(self):
        """Create a PDFParser instance for testing."""
        return PDFParser()
    
    @pytest.mark.asyncio
    async def test_extract_text_missing_library(self, pdf_parser):
        """Test handling of missing PyPDF2 library."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(DocumentProcessingError, match="PyPDF2 library not installed"):
                await pdf_parser.extract_text("/fake/path.pdf")


@pytest.mark.skipif(
    not pytest.importorskip("docx", reason="python-docx not available"),
    reason="python-docx not installed"
)
class TestWordParser:
    """Test cases for WordParser class (requires python-docx)."""
    
    @pytest.fixture
    def word_parser(self):
        """Create a WordParser instance for testing."""
        return WordParser()
    
    @pytest.mark.asyncio
    async def test_extract_text_missing_library(self, word_parser):
        """Test handling of missing python-docx library."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(DocumentProcessingError, match="python-docx library not installed"):
                await word_parser.extract_text("/fake/path.docx")


@pytest.mark.skipif(
    not pytest.importorskip("bs4", reason="beautifulsoup4 not available"),
    reason="beautifulsoup4 not installed"
)
class TestHTMLParser:
    """Test cases for HTMLParser class (requires beautifulsoup4)."""
    
    @pytest.fixture
    def html_parser(self):
        """Create an HTMLParser instance for testing."""
        return HTMLParser()
    
    @pytest.mark.asyncio
    async def test_extract_text_basic_html(self, html_parser):
        """Test extracting text from basic HTML."""
        html_content = """
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a paragraph.</p>
            <h2>Subheading</h2>
            <p>Another paragraph with <strong>bold</strong> text.</p>
        </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = f.name
        
        try:
            text = await html_parser.extract_text(temp_path)
            
            assert "Test Document" in text
            assert "Main Heading" in text
            assert "This is a paragraph" in text
            assert "Subheading" in text
            assert "bold" in text
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_extract_text_missing_library(self, html_parser):
        """Test handling of missing beautifulsoup4 library."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(DocumentProcessingError, match="beautifulsoup4 library not installed"):
                await html_parser.extract_text("/fake/path.html")


class TestIntegration:
    """Integration tests for document processing workflow."""
    
    @pytest.mark.asyncio
    async def test_full_text_processing_workflow(self):
        """Test the complete text processing workflow."""
        # Create a sample text file
        sample_text = """# Document Title
        
This is a sample document with **bold** text and *italic* text.

## Section 1
- Item 1
- Item 2
- Item 3

## Section 2
This section contains more content with multiple paragraphs.

The document has various formatting elements.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_text)
            temp_path = f.name
        
        try:
            # Create document instance
            document = Document(
                filename="test.md",
                original_filename="test.md",
                file_path=temp_path,
                file_size=len(sample_text.encode()),
                file_type="md",
                mime_type="text/markdown",
                knowledge_base_id="kb_1",
                uploaded_by="user_1"
            )
            
            # Process document
            processor = DocumentProcessor()
            text, metadata = await processor.process_document(temp_path, document)
            
            # Verify results
            assert text is not None
            assert len(text) > 0
            assert isinstance(metadata, dict)
            
            # Check that markdown was processed
            assert "Document Title" in text
            assert "Section 1" in text
            assert "Section 2" in text
            
            # Check metadata
            assert 'character_count' in metadata
            assert 'word_count' in metadata
            assert 'line_count' in metadata
            assert metadata['file_size_bytes'] == len(sample_text.encode())
            
        finally:
            os.unlink(temp_path)