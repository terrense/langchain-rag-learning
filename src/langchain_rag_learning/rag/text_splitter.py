"""Text splitting and chunking module for document processing."""

import logging  # Structured logging for debugging and monitoring
import re  # Regular expressions for text processing
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple  # Type hints for better code documentation

from langchain_rag_learning.core.exceptions import DocumentProcessingError  # LangChain framework for LLM applications
from langchain_rag_learning.core.models import Document, DocumentChunk  # LangChain framework for LLM applications

logger = logging.getLogger(__name__)


class BaseTextSplitter(ABC):
    """Base class for text splitting strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        keep_separator: bool = False
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            length_function: Function to calculate text length
            keep_separator: Whether to keep separators in chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        pass
    
    def create_document_chunks(
        self,
        text: str,
        document: Document,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Create DocumentChunk objects from text.
        
        Args:
            text: Input text to chunk
            document: Source document
            base_metadata: Base metadata to include in chunks
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = self.split_text(text)
        document_chunks = []
        
        current_position = 0
        
        for i, chunk_text in enumerate(chunks):
            # Find the actual position of this chunk in the original text
            chunk_start = text.find(chunk_text, current_position)
            if chunk_start == -1:
                # Fallback if exact match not found
                chunk_start = current_position
            
            chunk_end = chunk_start + len(chunk_text)
            
            # Prepare chunk metadata
            chunk_metadata = base_metadata.copy() if base_metadata else {}
            chunk_metadata.update({
                'chunk_method': self.__class__.__name__,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'source_document_id': document.id,
                'source_filename': document.filename
            })
            
            # Create DocumentChunk
            document_chunk = DocumentChunk(
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                start_char=chunk_start,
                end_char=chunk_end,
                metadata=chunk_metadata
            )
            
            document_chunks.append(document_chunk)
            
            # Update position for next chunk
            current_position = chunk_start + len(chunk_text) - self.chunk_overlap
        
        return document_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge splits into chunks of appropriate size.
        
        Args:
            splits: List of text splits
            separator: Separator used for splitting
            
        Returns:
            List of merged chunks
        """
        separator_len = self.length_function(separator)
        
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            split_len = self.length_function(split)
            
            if total + split_len + (separator_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    
                    # Start new chunk with overlap
                    while (
                        total > self.chunk_overlap
                        or (total + split_len + (separator_len if current_doc else 0) > self.chunk_size and total > 0)
                    ):
                        total -= self.length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            
            current_doc.append(split)
            total += split_len + (separator_len if len(current_doc) > 1 else 0)
        
        # Add remaining content
        if current_doc:
            doc = self._join_docs(current_doc, separator)
            if doc is not None:
                docs.append(doc)
        
        return docs
    
    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        """Join documents with separator."""
        text = separator.join(docs).strip()
        return text if text else None


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Recursively splits text using different separators in order of preference.
    Tries to keep related content together by using semantic separators.
    """
    
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        keep_separator: bool = False
    ):
        """
        Initialize recursive character text splitter.
        
        Args:
            separators: List of separators to try in order
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            length_function: Function to calculate text length
            keep_separator: Whether to keep separators in chunks
        """
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator)
        
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            " ",     # Spaces
            ""       # Character level
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split text recursively using different separators."""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        final_chunks = []
        
        # Get the separator to use
        separator = separators[-1] if separators else ""
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # Split the text
        splits = self._split_on_separator(text, separator)
        
        # Merge splits into appropriate chunks
        good_splits = []
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                
                if not new_separators:
                    final_chunks.append(split)
                else:
                    other_info = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(other_info)
        
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        
        return final_chunks
    
    def _split_on_separator(self, text: str, separator: str) -> List[str]:
        """Split text on separator."""
        if separator:
            if self.keep_separator:
                # Keep separator with the following text
                splits = text.split(separator)
                splits = [separator + s if i > 0 else s for i, s in enumerate(splits)]
                splits = [s for s in splits if s]
            else:
                splits = text.split(separator)
        else:
            splits = list(text)
        
        return [s for s in splits if s]


class SemanticTextSplitter(BaseTextSplitter):
    """
    Semantic-aware text splitter that tries to keep semantically related content together.
    Uses sentence boundaries and paragraph structure to create more coherent chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        sentence_split_regex: Optional[str] = None
    ):
        """
        Initialize semantic text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            length_function: Function to calculate text length
            sentence_split_regex: Custom regex for sentence splitting
        """
        super().__init__(chunk_size, chunk_overlap, length_function)
        
        # Default sentence splitting pattern
        self.sentence_pattern = sentence_split_regex or r'(?<=[.!?])\s+'
        
        # Patterns for different text structures
        self.paragraph_pattern = r'\n\s*\n'
        self.heading_pattern = r'^(#{1,6}\s+.+|.+\n[=-]+\s*$)'
        self.list_pattern = r'^[\s]*[-*+•]\s+|^\s*\d+\.\s+'
    
    def split_text(self, text: str) -> List[str]:
        """Split text using semantic boundaries."""
        # First, split by paragraphs
        paragraphs = re.split(self.paragraph_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this is a heading
            is_heading = bool(re.match(self.heading_pattern, paragraph, re.MULTILINE))
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if (current_chunk and 
                self.length_function(current_chunk + "\n\n" + paragraph) > self.chunk_size):
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                if self.length_function(paragraph) > self.chunk_size:
                    # Split large paragraph by sentences
                    sentence_chunks = self._split_paragraph_by_sentences(paragraph)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Apply overlap between chunks
        return self._apply_overlap(chunks)
    
    def _split_paragraph_by_sentences(self, paragraph: str) -> List[str]:
        """Split a large paragraph by sentences."""
        sentences = re.split(self.sentence_pattern, paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if (current_chunk and 
                self.length_function(current_chunk + " " + sentence) > self.chunk_size):
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks."""
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)
            
            if overlap_text:
                overlapped_chunk = overlap_text + "\n\n" + chunk
            else:
                overlapped_chunk = chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= overlap_size:
            return text
        
        # Try to find a good breaking point (sentence or paragraph boundary)
        overlap_start = len(text) - overlap_size
        
        # Look for sentence boundaries in the overlap region
        overlap_region = text[overlap_start:]
        sentence_matches = list(re.finditer(self.sentence_pattern, overlap_region))
        
        if sentence_matches:
            # Use the last sentence boundary
            last_boundary = sentence_matches[-1].end()
            return text[overlap_start + last_boundary:].strip()
        
        # Fallback to character-based overlap
        return text[-overlap_size:].strip()


class ContextAwareTextSplitter(BaseTextSplitter):
    """
    Context-aware text splitter that maintains context information across chunks.
    Adds contextual information like section headers to chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        include_context: bool = True,
        max_context_size: int = 200
    ):
        """
        Initialize context-aware text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            length_function: Function to calculate text length
            include_context: Whether to include contextual information
            max_context_size: Maximum size of context to include
        """
        super().__init__(chunk_size, chunk_overlap, length_function)
        self.include_context = include_context
        self.max_context_size = max_context_size
        
        # Patterns for identifying structure
        self.heading_patterns = [
            r'^(#{1,6}\s+.+)$',  # Markdown headers
            r'^(.+)\n[=-]+\s*$',  # Underlined headers
            r'^(\d+\.?\s+.+)$',   # Numbered sections
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split text while maintaining context."""
        # First identify document structure
        structure = self._analyze_document_structure(text)
        
        # Split text using semantic splitter as base
        semantic_splitter = SemanticTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function
        )
        
        base_chunks = semantic_splitter.split_text(text)
        
        if not self.include_context:
            return base_chunks
        
        # Add context to chunks
        contextualized_chunks = []
        
        for chunk in base_chunks:
            # Find the position of this chunk in the original text
            chunk_position = text.find(chunk)
            
            # Get relevant context for this chunk
            context = self._get_chunk_context(text, chunk_position, structure)
            
            # Add context to chunk if it fits
            if context and len(context) <= self.max_context_size:
                contextualized_chunk = f"{context}\n\n{chunk}"
                
                # Ensure the chunk doesn't exceed size limit
                if self.length_function(contextualized_chunk) <= self.chunk_size:
                    contextualized_chunks.append(contextualized_chunk)
                else:
                    contextualized_chunks.append(chunk)
            else:
                contextualized_chunks.append(chunk)
        
        return contextualized_chunks
    
    def _analyze_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Analyze document structure to identify sections and headers."""
        structure = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any heading pattern
            for pattern in self.heading_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    structure.append({
                        'type': 'heading',
                        'text': match.group(1),
                        'line_number': i,
                        'position': text.find(line)
                    })
                    break
        
        return structure
    
    def _get_chunk_context(
        self,
        text: str,
        chunk_position: int,
        structure: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Get relevant context for a chunk based on document structure."""
        if not structure:
            return None
        
        # Find the most recent heading before this chunk
        relevant_headings = []
        
        for item in structure:
            if item['position'] < chunk_position:
                relevant_headings.append(item)
        
        if not relevant_headings:
            return None
        
        # Get the most recent heading(s) as context
        context_parts = []
        
        # Include the most recent heading
        latest_heading = relevant_headings[-1]
        context_parts.append(f"Section: {latest_heading['text']}")
        
        # If there are multiple levels, include parent headings
        if len(relevant_headings) > 1:
            # Simple heuristic: if the latest heading is shorter, it might be a subsection
            prev_heading = relevant_headings[-2]
            if len(latest_heading['text']) > len(prev_heading['text']):
                context_parts.insert(0, f"Chapter: {prev_heading['text']}")
        
        return " | ".join(context_parts)


class DocumentChunker:
    """Main document chunking coordinator that manages different splitting strategies."""
    
    def __init__(self):
        """Initialize the document chunker with available strategies."""
        self.strategies = {
            'recursive': RecursiveCharacterTextSplitter,
            'semantic': SemanticTextSplitter,
            'context_aware': ContextAwareTextSplitter
        }
        self.default_strategy = 'semantic'
    
    async def chunk_document(
        self,
        text: str,
        document: Document,
        strategy: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Chunk a document using the specified strategy.
        
        Args:
            text: Text content to chunk
            document: Source document
            strategy: Chunking strategy to use
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            **kwargs: Additional parameters for the chunking strategy
            
        Returns:
            List of DocumentChunk objects
            
        Raises:
            DocumentProcessingError: If chunking fails
        """
        try:
            strategy = strategy or self.default_strategy
            
            if strategy not in self.strategies:
                raise DocumentProcessingError(f"Unknown chunking strategy: {strategy}")
            
            # Initialize the splitter
            splitter_class = self.strategies[strategy]
            splitter = splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
            
            # Create base metadata
            base_metadata = {
                'chunking_strategy': strategy,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            
            # Create chunks
            chunks = splitter.create_document_chunks(text, document, base_metadata)
            
            logger.info(f"Created {len(chunks)} chunks for document {document.filename}")
            return chunks
            
        except Exception as e:
            error_msg = f"Failed to chunk document {document.filename}: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get information about a chunking strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        strategy_class = self.strategies[strategy]
        
        return {
            'name': strategy,
            'class': strategy_class.__name__,
            'description': strategy_class.__doc__ or "No description available",
            'parameters': self._get_strategy_parameters(strategy_class)
        }
    
    def _get_strategy_parameters(self, strategy_class) -> Dict[str, Any]:
        """Extract parameter information from strategy class."""
        import inspect
        
        try:
            signature = inspect.signature(strategy_class.__init__)
            parameters = {}
            
            for name, param in signature.parameters.items():
                if name == 'self':
                    continue
                
                param_info = {
                    'type': param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
                
                parameters[name] = param_info
            
            return parameters
            
        except Exception:
            return {}