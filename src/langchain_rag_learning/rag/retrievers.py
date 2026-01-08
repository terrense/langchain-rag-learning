"""
Retrieval engines for RAG (Retrieval-Augmented Generation) system.

This module implements various retrieval strategies for finding relevant documents:
- Dense retrieval: Uses vector embeddings and semantic similarity
- Sparse retrieval: Uses keyword-based methods like BM25 and TF-IDF
- Hybrid retrieval: Combines dense and sparse methods with fusion algorithms
- Reranking: Improves results using cross-encoder models

Technical Overview:
- Dense retrieval leverages neural embeddings to capture semantic meaning
- Sparse retrieval uses traditional IR methods for exact keyword matching
- Hybrid approaches balance semantic understanding with keyword precision
- Reranking provides final optimization using sophisticated scoring models
"""

# Standard library imports for core functionality
import asyncio  # Asynchronous programming support for concurrent operations  # Async programming support for concurrent operations
import logging  # Logging framework for debugging and monitoring  # Structured logging for debugging and monitoring
import math     # Mathematical functions for scoring algorithms
import time     # Time utilities for performance measurement  # Time utilities for performance measurement
from abc import ABC, abstractmethod  # Abstract base classes for interface definition
from collections import Counter, defaultdict  # Data structures for counting and grouping  # Regular expressions for text processing
from typing import Any, Dict, List, Optional, Tuple, Union  # Type hints for better code clarity  # Type hints for better code documentation

# Third-party imports
import numpy as np  # Numerical computing library for vector operations  # Numerical computing library

# Internal imports from the RAG learning system
from langchain_rag_learning.core.config import get_settings  # Configuration management  # LangChain framework for LLM applications
from langchain_rag_learning.core.exceptions import DocumentProcessingError  # Custom exceptions  # LangChain framework for LLM applications
from langchain_rag_learning.core.models import DocumentChunk  # Data models  # LangChain framework for LLM applications
from langchain_rag_learning.rag.embeddings import EmbeddingManager, cosine_similarity  # Embedding utilities  # LangChain framework for LLM applications
from langchain_rag_learning.rag.vector_store import BaseVectorStore, VectorStoreManager  # Vector storage  # LangChain framework for LLM applications

# Initialize logger for this module - helps with debugging and monitoring
logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval implementations.
    
    This class defines the common interface that all retrievers must implement.
    It follows the Strategy pattern, allowing different retrieval algorithms
    to be used interchangeably.
    
    Design Pattern: Strategy Pattern
    - Encapsulates retrieval algorithms in separate classes
    - Allows runtime switching between different retrieval strategies
    - Promotes code reusability and maintainability
    """
    
    def __init__(self, name: str):
        """
        Initialize the base retriever with a unique name.
        
        Args:
            name (str): Unique identifier for this retriever instance.
                       Used for logging, debugging, and metadata tracking.
        
        Technical Note:
            The name is stored as an instance variable and used throughout
            the retrieval pipeline for identification and metadata purposes.
        """
        self.name = name  # Store retriever name for identification
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Abstract method for document retrieval - must be implemented by subclasses.
        
        This method defines the core retrieval interface that all concrete
        retriever implementations must provide.
        
        Args:
            query (str): The search query string from the user
            k (int, optional): Maximum number of documents to retrieve. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]], optional): 
                Metadata filters to apply during retrieval. Format: {"key": "value"}
            **kwargs: Additional retriever-specific parameters that vary by implementation
            
        Returns:
            List[Tuple[DocumentChunk, float]]: List of tuples containing:
                - DocumentChunk: The retrieved document chunk with content and metadata
                - float: Relevance score (higher = more relevant)
                
        Technical Notes:
            - Uses async/await for non-blocking I/O operations
            - Returns tuples for efficient packing of document and score
            - Scores should be normalized or comparable within retriever type
            - Filter_dict enables metadata-based filtering (e.g., by document type, date)
        """
        pass  # Must be implemented by concrete subclasses
    
    def get_name(self) -> str:
        """
        Get the name of this retriever instance.
        
        Returns:
            str: The retriever's unique name identifier
            
        Technical Note:
            Simple getter method following encapsulation principles.
            Provides controlled access to the internal name attribute.
        """
        return self.name


class DenseRetriever(BaseRetriever):
    """
    Dense retriever implementation using vector similarity search.
    
    This retriever uses neural embeddings to capture semantic meaning and find
    documents that are conceptually similar to the query, even if they don't
    share exact keywords.
    
    Technical Approach:
    - Converts queries and documents into high-dimensional vectors (embeddings)
    - Uses cosine similarity or other distance metrics for relevance scoring
    - Leverages pre-trained language models for semantic understanding
    - Excels at finding conceptually related content beyond keyword matching
    
    Use Cases:
    - Semantic search where meaning matters more than exact words
    - Cross-lingual retrieval with multilingual embeddings
    - Finding paraphrases and conceptually similar content
    - Handling synonyms and related concepts automatically
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_manager: EmbeddingManager,
        name: str = "dense_retriever"
    ):
        """
        Initialize the dense retriever with required components.
        
        Args:
            vector_store (BaseVectorStore): Storage system for document embeddings.
                Handles efficient similarity search operations (e.g., Chroma, FAISS).
            embedding_manager (EmbeddingManager): Manages embedding model operations.
                Converts text to vectors using various providers (OpenAI, HuggingFace, etc.).
            name (str, optional): Unique identifier for this retriever. 
                Defaults to "dense_retriever".
        
        Technical Notes:
            - Vector store must be pre-populated with document embeddings
            - Embedding manager should use the same model as document embeddings
            - Different embedding models produce incompatible vector spaces
        """
        super().__init__(name)  # Initialize parent class with retriever name
        self.vector_store = vector_store  # Store reference to vector database
        self.embedding_manager = embedding_manager  # Store reference to embedding service
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        embedding_provider: Optional[str] = None,
        similarity_threshold: float = 0.0,
        **kwargs
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve documents using dense vector similarity search.
        
        This method implements the core dense retrieval algorithm:
        1. Convert query text to embedding vector
        2. Search vector store for similar document embeddings
        3. Apply similarity threshold filtering
        4. Add retrieval metadata to results
        
        Args:
            query (str): The search query text to find similar documents for
            k (int, optional): Maximum number of documents to retrieve. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]], optional): 
                Metadata filters for document selection. Defaults to None.
            embedding_provider (Optional[str], optional): 
                Specific embedding provider to use. Defaults to None (uses default).
            similarity_threshold (float, optional): 
                Minimum similarity score for inclusion. Defaults to 0.0 (no filtering).
            **kwargs: Additional parameters passed to vector store
            
        Returns:
            List[Tuple[DocumentChunk, float]]: Retrieved documents with similarity scores,
            sorted by relevance (highest scores first)
            
        Raises:
            DocumentProcessingError: If embedding generation or vector search fails
            
        Technical Details:
            - Similarity scores typically range from 0.0 to 1.0 (cosine similarity)
            - Higher scores indicate greater semantic similarity
            - Threshold filtering helps remove low-quality matches
            - Metadata includes timing and scoring information for analysis
        """
        try:
            # Record start time for performance measurement
            start_time = time.time()
            
            # Convert query text to embedding vector using the embedding manager
            # This is the core of semantic search - transforming text to numerical representation
            query_embedding = await self.embedding_manager.embed_query(
                query, provider=embedding_provider
            )
            
            # Perform similarity search in the vector store
            # Vector store handles efficient nearest neighbor search algorithms
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,  # Query vector to match against
                k=k,  # Number of results to return
                filter_dict=filter_dict  # Optional metadata filtering
            )
            
            # Apply similarity threshold filtering if specified
            # This removes results below a minimum quality threshold
            if similarity_threshold > 0:
                results = [
                    (chunk, score) for chunk, score in results
                    if score >= similarity_threshold  # Keep only high-quality matches
                ]
            
            # Calculate total retrieval time for performance monitoring
            retrieval_time = time.time() - start_time
            
            # Add retrieval metadata to each result for debugging and analysis
            for chunk, score in results:
                # Initialize retrieval_info dictionary if it doesn't exist
                if 'retrieval_info' not in chunk.metadata:
                    chunk.metadata['retrieval_info'] = {}
                
                # Update metadata with retrieval information
                chunk.metadata['retrieval_info'].update({
                    'retriever': self.name,  # Which retriever was used
                    'similarity_score': score,  # Numerical similarity score
                    'retrieval_time': retrieval_time,  # Time taken for retrieval
                    'query': query[:100]  # Store truncated query for reference
                })
            
            # Log successful retrieval for monitoring
            logger.info(f"Dense retrieval found {len(results)} documents in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            # Convert any exception to our custom exception type with context
            raise DocumentProcessingError(f"Dense retrieval failed: {str(e)}")
    
    async def get_query_expansion_terms(
        self,
        query: str,
        expansion_count: int = 5
    ) -> List[str]:
        """
        Generate query expansion terms using semantic similarity.
        
        Query expansion improves retrieval by adding related terms to the original query.
        This method uses the semantic understanding of dense retrieval to find
        conceptually related terms from top-retrieved documents.
        
        Algorithm:
        1. Perform initial retrieval with original query
        2. Extract text from top-ranked results
        3. Analyze term frequency in retrieved content
        4. Filter out common stop words and original query terms
        5. Return most frequent meaningful terms
        
        Args:
            query (str): Original query string to expand
            expansion_count (int, optional): Number of expansion terms to generate. 
                Defaults to 5.
        
        Returns:
            List[str]: List of expansion terms that can enhance the original query
            
        Technical Notes:
            - Uses simple frequency-based term extraction (can be enhanced with NLP)
            - Filters common stop words to focus on meaningful content words
            - Could be improved with techniques like pseudo-relevance feedback
            - Expansion terms should be semantically related to original query
        """
        try:
            # Get initial retrieval results to analyze for expansion terms
            initial_results = await self.retrieve(query, k=10)
            
            # Return empty list if no results found
            if not initial_results:
                return []
            
            # Extract and combine text from top results for term analysis
            # Focus on top 3 results as they're most likely to be relevant
            all_text = " ".join([chunk.content for chunk, _ in initial_results[:3]])
            
            # Simple tokenization: split text into words and convert to lowercase
            # This could be enhanced with proper NLP tokenization
            words = all_text.lower().split()
            
            # Count frequency of each word for importance ranking
            word_freq = Counter(words)
            
            # Define common stop words to filter out
            # These words are too common to be useful for expansion
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # Filter and select expansion terms based on criteria:
            # - Not in common stop words
            # - Length > 2 characters (avoid short, meaningless words)
            # - Not already in original query (avoid redundancy)
            expansion_terms = [
                word for word, freq in word_freq.most_common(expansion_count * 2)
                if word not in common_words and len(word) > 2 and word not in query.lower()
            ]
            
            # Return top expansion terms up to requested count
            return expansion_terms[:expansion_count]
            
        except Exception as e:
            # Log warning but don't fail - expansion is optional enhancement
            logger.warning(f"Query expansion failed: {e}")
            return []  # Return empty list on failure


class SparseRetriever(BaseRetriever):
    """
    Sparse retriever implementation using keyword-based search algorithms.
    
    This retriever uses traditional Information Retrieval (IR) methods like BM25 and TF-IDF
    to find documents based on exact keyword matches and term frequency statistics.
    
    Technical Approach:
    - Builds inverted index mapping terms to documents containing them
    - Uses statistical scoring functions (BM25, TF-IDF) for relevance calculation
    - Excels at finding documents with specific keywords and phrases
    - Provides interpretable scoring based on term frequency and document frequency
    
    Algorithms Supported:
    - BM25: Best Matching 25, probabilistic ranking function
    - TF-IDF: Term Frequency-Inverse Document Frequency weighting
    
    Use Cases:
    - Exact keyword matching and phrase search
    - Domain-specific terminology retrieval
    - When interpretability of results is important
    - Complementing semantic search with keyword precision
    """
    
    def __init__(
        self,
        documents: List[DocumentChunk],
        name: str = "sparse_retriever",
        k1: float = 1.2,
        b: float = 0.75
    ):
        """
        Initialize sparse retriever with document collection and BM25 parameters.
        
        Args:
            documents (List[DocumentChunk]): Collection of document chunks to index.
                These will be tokenized and indexed for efficient keyword search.
            name (str, optional): Unique identifier for this retriever. 
                Defaults to "sparse_retriever".
            k1 (float, optional): BM25 term frequency saturation parameter. 
                Controls how quickly term frequency scores saturate. Defaults to 1.2.
            b (float, optional): BM25 document length normalization parameter.
                Controls how much document length affects scoring (0=no effect, 1=full effect).
                Defaults to 0.75.
        
        Technical Notes:
            - k1 typically ranges from 1.0 to 2.0 (1.2 is standard)
            - b typically ranges from 0.0 to 1.0 (0.75 is standard)
            - Higher k1 values give more weight to term frequency
            - Higher b values penalize longer documents more strongly
        """
        super().__init__(name)  # Initialize parent class
        self.documents = documents  # Store document collection
        self.k1 = k1  # BM25 term frequency parameter
        self.b = b    # BM25 document length normalization parameter
        
        # Initialize data structures for inverted index
        self.inverted_index = {}  # Maps terms to {doc_id: term_frequency}
        self.doc_lengths = {}     # Maps doc_id to document length in tokens
        self.avg_doc_length = 0   # Average document length across collection
        self.doc_frequencies = {} # Maps terms to number of documents containing them
        self.total_docs = len(documents)  # Total number of documents in collection
        
        # Build the inverted index immediately upon initialization
        self._build_index()
    
    def _build_index(self):
        """
        Build inverted index for efficient keyword-based retrieval.
        
        The inverted index is the core data structure for sparse retrieval:
        - Maps each unique term to the documents containing it
        - Stores term frequencies for scoring calculations
        - Calculates document statistics needed for BM25 scoring
        
        Index Structure:
        - inverted_index: {term: {doc_id: frequency}}
        - doc_lengths: {doc_id: token_count}
        - doc_frequencies: {term: document_count}
        
        Technical Process:
        1. Tokenize each document into terms
        2. Count term frequencies within each document
        3. Build inverted mappings from terms to documents
        4. Calculate collection-wide statistics (avg length, doc frequencies)
        """
        logger.info(f"Building sparse index for {len(self.documents)} documents")
        
        # Track total length across all documents for average calculation
        total_length = 0
        
        # Process each document to build index
        for i, doc in enumerate(self.documents):
            # Tokenize document content into individual terms
            # This converts raw text into searchable tokens
            tokens = self._tokenize(doc.content)
            doc_length = len(tokens)
            
            # Store document length for BM25 normalization
            self.doc_lengths[i] = doc_length
            total_length += doc_length
            
            # Count frequency of each term in this document
            # Counter creates {term: frequency} mapping efficiently
            term_freq = Counter(tokens)
            
            # Update inverted index with term frequencies
            for term, freq in term_freq.items():
                # Initialize term entry if first occurrence
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                
                # Store term frequency for this document
                self.inverted_index[term][i] = freq
        
        # Calculate average document length for BM25 normalization
        # Prevents bias toward longer or shorter documents
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
        
        # Calculate document frequencies (DF) for each term
        # DF = number of documents containing the term (used in IDF calculation)
        for term, doc_dict in self.inverted_index.items():
            self.doc_frequencies[term] = len(doc_dict)
        
        logger.info(f"Built index with {len(self.inverted_index)} unique terms")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into searchable terms using simple regex-based approach.
        
        This method converts raw text into a list of normalized tokens suitable
        for keyword matching. The tokenization strategy significantly impacts
        retrieval quality and should match the query processing approach.
        
        Args:
            text (str): Raw text content to tokenize
            
        Returns:
            List[str]: List of normalized tokens (lowercase, filtered)
            
        Tokenization Process:
        1. Convert to lowercase for case-insensitive matching
        2. Extract word boundaries using regex pattern
        3. Filter out very short tokens (< 3 characters)
        4. Remove punctuation and special characters
        
        Technical Notes:
        - Uses regex pattern r'\b\w+\b' to find word boundaries
        - Filters tokens shorter than 3 characters to reduce noise
        - Could be enhanced with stemming, lemmatization, or language-specific rules
        - Should be consistent with query tokenization for accurate matching
        """
        # Import regex module for pattern matching
        import re  # Regular expressions for text processing
        
        # Extract word tokens using regex pattern
        # \b = word boundary, \w+ = one or more word characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out short tokens that are typically not meaningful
        # Length > 2 removes articles, prepositions, and abbreviations
        return [token for token in tokens if len(token) > 2]
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_index: int) -> float:
        """
        Calculate BM25 relevance score for a document given query terms.
        
        BM25 (Best Matching 25) is a probabilistic ranking function that estimates
        the relevance of documents to a given search query. It's widely used in
        search engines and considered state-of-the-art for keyword-based retrieval.
        
        BM25 Formula:
        score = Σ(IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl)))
        
        Where:
        - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
        - f(qi,D) = frequency of term qi in document D
        - |D| = length of document D in tokens
        - avgdl = average document length in collection
        - k1, b = tuning parameters
        
        Args:
            query_terms (List[str]): List of tokenized query terms
            doc_index (int): Index of document to score
            
        Returns:
            float: BM25 relevance score (higher = more relevant)
            
        Technical Details:
        - IDF component reduces weight of common terms
        - TF component increases weight of frequent terms with saturation
        - Length normalization prevents bias toward long/short documents
        - Smoothing constants (0.5) prevent division by zero
        """
        score = 0.0  # Initialize cumulative score
        doc_length = self.doc_lengths.get(doc_index, 0)  # Get document length
        
        # Calculate score contribution for each query term
        for term in query_terms:
            # Check if term exists in document
            if term in self.inverted_index and doc_index in self.inverted_index[term]:
                # Get term frequency in this document
                tf = self.inverted_index[term][doc_index]
                
                # Get document frequency (number of docs containing term)
                df = self.doc_frequencies[term]
                
                # Calculate Inverse Document Frequency (IDF)
                # Higher IDF for rare terms, lower for common terms
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))
                
                # Calculate BM25 term frequency component with saturation
                # Numerator: tf * (k1 + 1) - gives weight to term frequency
                numerator = tf * (self.k1 + 1)
                
                # Denominator: tf + k1 * (1 - b + b * doc_length/avg_length)
                # Provides term frequency saturation and length normalization
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                # Add this term's contribution to total score
                score += idf * (numerator / denominator)
        
        return score
    
    def _calculate_tfidf_score(self, query_terms: List[str], doc_index: int) -> float:
        """
        Calculate TF-IDF relevance score for a document given query terms.
        
        TF-IDF (Term Frequency-Inverse Document Frequency) is a classical
        information retrieval scoring method that balances term importance
        within documents against term rarity in the collection.
        
        TF-IDF Formula:
        score = Σ(TF(qi,D) * IDF(qi))
        
        Where:
        - TF(qi,D) = (frequency of qi in D) / (total terms in D)
        - IDF(qi) = log(total_documents / documents_containing_qi)
        
        Args:
            query_terms (List[str]): List of tokenized query terms
            doc_index (int): Index of document to score
            
        Returns:
            float: TF-IDF relevance score (higher = more relevant)
            
        Technical Details:
        - TF component: normalized term frequency within document
        - IDF component: inverse document frequency across collection
        - Simpler than BM25 but less sophisticated in handling term saturation
        - Good baseline method for keyword-based retrieval
        """
        score = 0.0  # Initialize cumulative score
        doc_length = self.doc_lengths.get(doc_index, 0)  # Get document length
        
        # Calculate score contribution for each query term
        for term in query_terms:
            # Check if term exists in document
            if term in self.inverted_index and doc_index in self.inverted_index[term]:
                # Calculate normalized Term Frequency (TF)
                # Dividing by doc_length normalizes for document length
                tf = self.inverted_index[term][doc_index] / doc_length if doc_length > 0 else 0
                
                # Get document frequency for IDF calculation
                df = self.doc_frequencies[term]
                
                # Calculate Inverse Document Frequency (IDF)
                # log(N/df) - higher for rare terms, lower for common terms
                idf = math.log(self.total_docs / df) if df > 0 else 0
                
                # Add TF * IDF contribution to total score
                score += tf * idf
        
        return score
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        scoring_method: str = "bm25",
        query_expansion: bool = False,
        **kwargs
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve documents using sparse keyword-based search methods.
        
        This method implements the main sparse retrieval algorithm:
        1. Tokenize the input query into searchable terms
        2. Optionally expand query with related terms
        3. Score all documents using selected algorithm (BM25 or TF-IDF)
        4. Apply metadata filters if specified
        5. Sort results by relevance score and return top-k
        
        Args:
            query (str): Search query string to find matching documents
            k (int, optional): Maximum number of documents to retrieve. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]], optional): 
                Metadata filters for document selection. Defaults to None.
            scoring_method (str, optional): Scoring algorithm to use.
                Options: "bm25" (default) or "tfidf". Defaults to "bm25".
            query_expansion (bool, optional): Whether to expand query with related terms.
                Defaults to False.
            **kwargs: Additional parameters for future extensions
            
        Returns:
            List[Tuple[DocumentChunk, float]]: Retrieved documents with relevance scores,
            sorted by score in descending order (most relevant first)
            
        Raises:
            DocumentProcessingError: If retrieval process fails
            ValueError: If unknown scoring method is specified
            
        Technical Notes:
        - BM25 generally performs better than TF-IDF for most use cases
        - Query expansion can improve recall but may reduce precision
        - Metadata filtering is applied after scoring for efficiency
        - Scores are not normalized between different scoring methods
        """
        try:
            # Record start time for performance measurement
            start_time = time.time()
            
            # Tokenize query using same method as document indexing
            # Consistency in tokenization is crucial for accurate matching
            query_terms = self._tokenize(query)
            
            # Return empty results if no valid query terms
            if not query_terms:
                return []
            
            # Perform query expansion if requested
            # This adds related terms to improve recall
            if query_expansion:
                expanded_terms = await self._expand_query(query_terms)
                query_terms.extend(expanded_terms)
                query_terms = list(set(query_terms))  # Remove duplicates using set conversion
            
            # Initialize list to store document scores
            doc_scores = []
            
            # Score each document in the collection
            for i, doc in enumerate(self.documents):
                # Apply metadata filters first to skip irrelevant documents
                if filter_dict:
                    match = True
                    # Check each filter condition
                    for key, value in filter_dict.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            match = False
                            break  # Exit early if any condition fails
                    
                    # Skip this document if it doesn't match filters
                    if not match:
                        continue
                
                # Calculate relevance score based on selected method
                if scoring_method == "bm25":
                    score = self._calculate_bm25_score(query_terms, i)
                elif scoring_method == "tfidf":
                    score = self._calculate_tfidf_score(query_terms, i)
                else:
                    raise ValueError(f"Unknown scoring method: {scoring_method}")
                
                # Only include documents with positive scores (contain query terms)
                if score > 0:
                    doc_scores.append((doc, score))
            
            # Sort documents by score in descending order (highest scores first)
            # Lambda function extracts score (second element) for sorting
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take only top-k results
            results = doc_scores[:k]
            
            # Calculate total retrieval time for performance monitoring
            retrieval_time = time.time() - start_time
            
            # Add retrieval metadata to each result for analysis and debugging
            for chunk, score in results:
                # Initialize retrieval_info if not present
                if 'retrieval_info' not in chunk.metadata:
                    chunk.metadata['retrieval_info'] = {}
                
                # Store comprehensive retrieval information
                chunk.metadata['retrieval_info'].update({
                    'retriever': self.name,  # Retriever identification
                    'relevance_score': score,  # Numerical relevance score
                    'scoring_method': scoring_method,  # Algorithm used
                    'retrieval_time': retrieval_time,  # Performance timing
                    'query_terms': query_terms,  # Processed query terms
                    'query': query[:100]  # Original query (truncated)
                })
            
            # Log successful retrieval for monitoring
            logger.info(f"Sparse retrieval found {len(results)} documents in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            # Convert any exception to our custom exception type
            raise DocumentProcessingError(f"Sparse retrieval failed: {str(e)}")
    
    async def _expand_query(self, query_terms: List[str]) -> List[str]:
        """
        Expand query with related terms using co-occurrence analysis.
        
        This method implements a simple query expansion technique based on
        term co-occurrence patterns in the document collection. It finds
        terms that frequently appear together with the original query terms.
        
        Algorithm:
        1. For each query term, find all documents containing it
        2. Extract all other terms from those documents
        3. Count co-occurrence frequencies
        4. Return most frequently co-occurring terms
        
        Args:
            query_terms (List[str]): Original tokenized query terms
            
        Returns:
            List[str]: List of expansion terms to add to the query
            
        Technical Notes:
        - Uses simple co-occurrence counting (could be enhanced with PMI, etc.)
        - Limits to top 3 expansion terms to avoid query drift
        - Excludes original query terms to avoid redundancy
        - Could be improved with more sophisticated relevance feedback methods
        """
        expansion_terms = []
        
        # Dictionary to count term co-occurrences
        # defaultdict automatically initializes missing keys to 0
        term_cooccurrence = defaultdict(int)
        
        # Analyze co-occurrence for each query term
        for term in query_terms:
            # Check if term exists in our index
            if term in self.inverted_index:
                # Get all documents containing this term
                docs_with_term = self.inverted_index[term].keys()
                
                # Analyze other terms in these documents
                for doc_idx in docs_with_term:
                    # Tokenize document content to find co-occurring terms
                    doc_tokens = self._tokenize(self.documents[doc_idx].content)
                    
                    # Count occurrences of other terms
                    for token in doc_tokens:
                        # Exclude the original term and existing query terms
                        if token != term and token not in query_terms:
                            term_cooccurrence[token] += 1
        
        # Select top co-occurring terms as expansion candidates
        # Sort by frequency and take top 3 to avoid over-expansion
        expansion_terms = [
            term for term, count in sorted(
                term_cooccurrence.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]  # Limit to top 3 expansion terms
        ]
        
        return expansion_terms
    
    def update_index(self, new_documents: List[DocumentChunk]):
        """
        Update the inverted index with new documents.
        
        This method allows dynamic addition of new documents to the retrieval
        system without rebuilding the entire index from scratch. However,
        for simplicity, it currently rebuilds the complete index.
        
        Args:
            new_documents (List[DocumentChunk]): New document chunks to add
            
        Technical Notes:
        - Currently rebuilds entire index for simplicity
        - Could be optimized for incremental updates in production systems
        - Maintains consistency of all index statistics after update
        - Logs the number of new documents added for monitoring
        """
        # Add new documents to existing collection
        self.documents.extend(new_documents)
        
        # Update total document count
        self.total_docs = len(self.documents)
        
        # Rebuild entire index (could be optimized for incremental updates)
        self._build_index()
        
        # Log successful update for monitoring
        logger.info(f"Updated sparse index with {len(new_documents)} new documents")


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines dense and sparse retrieval methods.
    
    This retriever leverages the strengths of both dense (semantic) and sparse (keyword)
    retrieval approaches to provide more comprehensive and robust search results.
    It implements various fusion algorithms to combine results from both methods.
    
    Technical Approach:
    - Executes dense and sparse retrieval in parallel for efficiency
    - Applies fusion algorithms to combine and rank results
    - Balances semantic understanding with keyword precision
    - Provides configurable weighting between retrieval methods
    
    Fusion Methods Supported:
    - RRF (Reciprocal Rank Fusion): Rank-based combination method
    - Weighted: Score-based combination with configurable weights
    - Max: Takes maximum score from either method
    
    Benefits:
    - Improved recall by combining different retrieval paradigms
    - Robustness against query variations and document types
    - Flexibility to adjust balance between semantic and keyword matching
    - Better performance across diverse query types and domains
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        name: str = "hybrid_retriever",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever with component retrievers and fusion weights.
        
        Args:
            dense_retriever (DenseRetriever): Dense retriever instance for semantic search
            sparse_retriever (SparseRetriever): Sparse retriever instance for keyword search
            name (str, optional): Unique identifier for this retriever. 
                Defaults to "hybrid_retriever".
            dense_weight (float, optional): Weight for dense retrieval scores in fusion.
                Should be between 0.0 and 1.0. Defaults to 0.7.
            sparse_weight (float, optional): Weight for sparse retrieval scores in fusion.
                Should be between 0.0 and 1.0. Defaults to 0.3.
        
        Technical Notes:
            - Weights should sum to 1.0 for proper normalization
            - Higher dense_weight favors semantic similarity
            - Higher sparse_weight favors keyword matching
            - Default 0.7/0.3 split generally works well for most use cases
        """
        super().__init__(name)  # Initialize parent class
        self.dense_retriever = dense_retriever    # Store dense retriever reference
        self.sparse_retriever = sparse_retriever  # Store sparse retriever reference
        self.dense_weight = dense_weight          # Weight for dense scores
        self.sparse_weight = sparse_weight        # Weight for sparse scores
        
        # Validate that weights sum to approximately 1.0
        # Use small epsilon (1e-6) to handle floating-point precision issues
        if abs(dense_weight + sparse_weight - 1.0) > 1e-6:
            logger.warning(
                f"Weights don't sum to 1.0: {dense_weight} + {sparse_weight} = "
                f"{dense_weight + sparse_weight}"
            )
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        normalize_scores: bool = True,
        **kwargs
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve documents using hybrid approach with configurable fusion methods.
        
        This method implements the core hybrid retrieval algorithm:
        1. Execute dense and sparse retrieval in parallel
        2. Apply selected fusion method to combine results
        3. Return top-k fused results with metadata
        
        Args:
            query (str): Search query string
            k (int, optional): Number of final results to return. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]], optional): 
                Metadata filters applied to both retrievers. Defaults to None.
            fusion_method (str, optional): Method for combining results.
                Options: "rrf", "weighted", "max". Defaults to "rrf".
            rrf_k (int, optional): Parameter for RRF fusion algorithm. 
                Higher values reduce rank differences. Defaults to 60.
            normalize_scores (bool, optional): Whether to normalize scores before fusion.
                Recommended for weighted and max fusion. Defaults to True.
            **kwargs: Additional parameters passed to component retrievers
            
        Returns:
            List[Tuple[DocumentChunk, float]]: Fused results sorted by combined score
            
        Raises:
            DocumentProcessingError: If retrieval or fusion process fails
            ValueError: If unknown fusion method is specified
            
        Technical Details:
        - Retrieves k*2 results from each method to ensure sufficient candidates
        - Parallel execution improves performance significantly
        - Different fusion methods have different characteristics:
          * RRF: Robust, rank-based, good for different score scales
          * Weighted: Score-based, requires score normalization
          * Max: Takes best score from either method
        """
        try:
            # Record start time for performance measurement
            start_time = time.time()
            
            # Execute both retrievers in parallel for efficiency
            # asyncio.create_task() enables concurrent execution
            # Retrieve more results (k*2) to have sufficient candidates for fusion
            dense_task = asyncio.create_task(
                self.dense_retriever.retrieve(
                    query, k=k*2, filter_dict=filter_dict, **kwargs
                )
            )
            sparse_task = asyncio.create_task(
                self.sparse_retriever.retrieve(
                    query, k=k*2, filter_dict=filter_dict, **kwargs
                )
            )
            
            # Wait for both retrievers to complete and get results
            # asyncio.gather() waits for all tasks and returns results in order
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            
            # Apply selected fusion method to combine results
            if fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, rrf_k)
            elif fusion_method == "weighted":
                fused_results = self._weighted_fusion(dense_results, sparse_results, normalize_scores)
            elif fusion_method == "max":
                fused_results = self._max_fusion(dense_results, sparse_results, normalize_scores)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            # Take only top-k results from fused list
            final_results = fused_results[:k]
            
            # Calculate total retrieval time including fusion
            retrieval_time = time.time() - start_time
            
            # Add comprehensive metadata to results for analysis
            for chunk, score in final_results:
                # Initialize retrieval_info if not present
                if 'retrieval_info' not in chunk.metadata:
                    chunk.metadata['retrieval_info'] = {}
                
                # Store hybrid retrieval metadata
                chunk.metadata['retrieval_info'].update({
                    'retriever': self.name,           # Hybrid retriever identification
                    'fusion_method': fusion_method,   # Fusion algorithm used
                    'hybrid_score': score,            # Final fused score
                    'retrieval_time': retrieval_time, # Total processing time
                    'dense_weight': self.dense_weight,   # Dense component weight
                    'sparse_weight': self.sparse_weight, # Sparse component weight
                    'query': query[:100]              # Original query (truncated)
                })
            
            # Log successful hybrid retrieval
            logger.info(f"Hybrid retrieval found {len(final_results)} documents in {retrieval_time:.3f}s")
            return final_results
            
        except Exception as e:
            # Convert any exception to our custom exception type
            raise DocumentProcessingError(f"Hybrid retrieval failed: {str(e)}")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        k: int = 60
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform Reciprocal Rank Fusion (RRF) to combine ranked lists.
        
        RRF is a robust fusion method that combines ranked lists without requiring
        score normalization. It's particularly effective when the component
        retrievers use different scoring scales or distributions.
        
        RRF Formula:
        RRF_score = Σ(1 / (k + rank_i))
        
        Where:
        - k is a constant (typically 60) that controls the impact of rank differences
        - rank_i is the rank of the document in retriever i (1-indexed)
        - Documents not appearing in a list get rank = infinity (score = 0)
        
        Args:
            dense_results (List[Tuple[DocumentChunk, float]]): Results from dense retriever
            sparse_results (List[Tuple[DocumentChunk, float]]): Results from sparse retriever
            k (int, optional): RRF parameter controlling rank impact. Defaults to 60.
            
        Returns:
            List[Tuple[DocumentChunk, float]]: Fused results sorted by RRF score
            
        Technical Notes:
        - RRF is robust to different score scales and distributions
        - Higher k values reduce the impact of rank differences
        - Documents appearing in both lists get higher scores
        - Method is symmetric and doesn't favor either retriever
        """
        # Create rank mappings for both result lists
        # Rank is 1-indexed (first result has rank 1)
        dense_ranks = {chunk.id: rank + 1 for rank, (chunk, _) in enumerate(dense_results)}
        sparse_ranks = {chunk.id: rank + 1 for rank, (chunk, _) in enumerate(sparse_results)}
        
        # Collect all unique document chunks from both result sets
        all_chunks = {}
        for chunk, _ in dense_results + sparse_results:
            all_chunks[chunk.id] = chunk  # Use chunk.id as key to avoid duplicates
        
        # Calculate RRF scores for all unique documents
        rrf_scores = {}
        for chunk_id, chunk in all_chunks.items():
            rrf_score = 0.0
            
            # Add RRF contribution from dense retriever if document appears there
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[chunk_id])
            
            # Add RRF contribution from sparse retriever if document appears there
            if chunk_id in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[chunk_id])
            
            # Store final RRF score for this document
            rrf_scores[chunk_id] = rrf_score
        
        # Sort documents by RRF score in descending order
        # Higher RRF scores indicate better combined relevance
        sorted_results = [
            (all_chunks[chunk_id], score)
            for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return sorted_results
    
    def _weighted_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        normalize_scores: bool = True
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform weighted score fusion.
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            normalize_scores: Whether to normalize scores
            
        Returns:
            Fused results sorted by weighted score
        """
        # Normalize scores if requested
        if normalize_scores:
            dense_results = self._normalize_scores(dense_results)
            sparse_results = self._normalize_scores(sparse_results)
        
        # Create score mappings
        dense_scores = {chunk.id: score for chunk, score in dense_results}
        sparse_scores = {chunk.id: score for chunk, score in sparse_results}
        
        # Collect all unique chunks
        all_chunks = {}
        for chunk, _ in dense_results + sparse_results:
            all_chunks[chunk.id] = chunk
        
        # Calculate weighted scores
        weighted_scores = {}
        for chunk_id, chunk in all_chunks.items():
            weighted_score = 0.0
            
            if chunk_id in dense_scores:
                weighted_score += self.dense_weight * dense_scores[chunk_id]
            
            if chunk_id in sparse_scores:
                weighted_score += self.sparse_weight * sparse_scores[chunk_id]
            
            weighted_scores[chunk_id] = weighted_score
        
        # Sort by weighted score
        sorted_results = [
            (all_chunks[chunk_id], score)
            for chunk_id, score in sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return sorted_results
    
    def _max_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        normalize_scores: bool = True
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform max score fusion.
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            normalize_scores: Whether to normalize scores
            
        Returns:
            Fused results sorted by max score
        """
        # Normalize scores if requested
        if normalize_scores:
            dense_results = self._normalize_scores(dense_results)
            sparse_results = self._normalize_scores(sparse_results)
        
        # Create score mappings
        dense_scores = {chunk.id: score for chunk, score in dense_results}
        sparse_scores = {chunk.id: score for chunk, score in sparse_results}
        
        # Collect all unique chunks
        all_chunks = {}
        for chunk, _ in dense_results + sparse_results:
            all_chunks[chunk.id] = chunk
        
        # Calculate max scores
        max_scores = {}
        for chunk_id, chunk in all_chunks.items():
            scores = []
            
            if chunk_id in dense_scores:
                scores.append(dense_scores[chunk_id])
            
            if chunk_id in sparse_scores:
                scores.append(sparse_scores[chunk_id])
            
            max_scores[chunk_id] = max(scores) if scores else 0.0
        
        # Sort by max score
        sorted_results = [
            (all_chunks[chunk_id], score)
            for chunk_id, score in sorted(max_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return sorted_results
    
    def _normalize_scores(
        self,
        results: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            results: List of (chunk, score) tuples
            
        Returns:
            Normalized results
        """
        if not results:
            return results
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            return [(chunk, 1.0) for chunk, _ in results]
        
        # Min-max normalization
        normalized_results = [
            (chunk, (score - min_score) / (max_score - min_score))
            for chunk, score in results
        ]
        
        return normalized_results
    
    def update_weights(self, dense_weight: float, sparse_weight: float):
        """
        Update fusion weights.
        
        Args:
            dense_weight: New dense weight
            sparse_weight: New sparse weight
        """
        if abs(dense_weight + sparse_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0: {dense_weight} + {sparse_weight}")
        
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        logger.info(f"Updated hybrid weights: dense={dense_weight}, sparse={sparse_weight}")


class Reranker:
    """Reranker for improving retrieval results using cross-encoder models."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to run model on
            batch_size: Batch size for reranking
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self._model_loaded = False
    
    async def _load_model(self):
        """Load the cross-encoder model."""
        if self._model_loaded:
            return
        
        try:
            from sentence_transformers import CrossEncoder  # HuggingFace transformers for NLP models
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(self.model_name, device=self.device)
            )
            
            self._model_loaded = True
            logger.info(f"Loaded reranker model: {self.model_name}")
            
        except ImportError:
            raise DocumentProcessingError(
                "sentence-transformers library not installed. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}")
            self._model_loaded = False
    
    async def rerank(
        self,
        query: str,
        results: List[Tuple[DocumentChunk, float]],
        top_k: Optional[int] = None,
        diversity_penalty: float = 0.0,
        novelty_penalty: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: Original query
            results: List of (chunk, score) tuples to rerank
            top_k: Number of top results to return
            diversity_penalty: Penalty for similar documents
            novelty_penalty: Penalty for common information
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        try:
            await self._load_model()
            
            if not self._model_loaded:
                # Fallback to original scores if model loading failed
                logger.warning("Reranker model not available, returning original results")
                return results[:top_k] if top_k else results
            
            start_time = time.time()
            
            # Prepare query-document pairs
            query_doc_pairs = [(query, chunk.content) for chunk, _ in results]
            
            # Get reranking scores in batches
            loop = asyncio.get_event_loop()
            
            async def _predict_batch(pairs_batch):
                """
                Async  predict batch function implementation.
                """
                return await loop.run_in_executor(
                    None,
                    lambda: self.model.predict(pairs_batch)
                )
            
            all_scores = []
            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch = query_doc_pairs[i:i + self.batch_size]
                batch_scores = await _predict_batch(batch)
                all_scores.extend(batch_scores)
            
            # Combine with original results
            reranked_results = [
                (chunk, float(score)) for (chunk, _), score in zip(results, all_scores)
            ]
            
            # Apply diversity and novelty penalties if specified
            if diversity_penalty > 0 or novelty_penalty > 0:
                reranked_results = self._apply_diversity_novelty_penalties(
                    reranked_results, diversity_penalty, novelty_penalty
                )
            
            # Sort by reranked scores
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            # Add reranking metadata
            reranking_time = time.time() - start_time
            
            for chunk, score in reranked_results:
                if 'retrieval_info' not in chunk.metadata:
                    chunk.metadata['retrieval_info'] = {}
                
                chunk.metadata['retrieval_info'].update({
                    'reranked': True,
                    'rerank_score': score,
                    'rerank_model': self.model_name,
                    'reranking_time': reranking_time
                })
            
            final_results = reranked_results[:top_k] if top_k else reranked_results
            
            logger.info(f"Reranked {len(results)} results in {reranking_time:.3f}s")
            return final_results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original results")
            return results[:top_k] if top_k else results
    
    def _apply_diversity_novelty_penalties(
        self,
        results: List[Tuple[DocumentChunk, float]],
        diversity_penalty: float,
        novelty_penalty: float
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Apply diversity and novelty penalties to results.
        
        Args:
            results: List of (chunk, score) tuples
            diversity_penalty: Penalty for similar documents
            novelty_penalty: Penalty for common information
            
        Returns:
            Results with penalties applied
        """
        if not results or (diversity_penalty == 0 and novelty_penalty == 0):
            return results
        
        penalized_results = []
        selected_contents = []
        
        for chunk, score in results:
            penalty = 0.0
            
            # Diversity penalty: penalize similar content
            if diversity_penalty > 0 and selected_contents:
                max_similarity = 0.0
                for selected_content in selected_contents:
                    # Simple similarity based on common words
                    similarity = self._calculate_content_similarity(chunk.content, selected_content)
                    max_similarity = max(max_similarity, similarity)
                
                penalty += diversity_penalty * max_similarity
            
            # Novelty penalty: penalize common terms
            if novelty_penalty > 0:
                # Simple novelty based on term frequency
                words = chunk.content.lower().split()
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                common_ratio = sum(1 for word in words if word in common_words) / len(words) if words else 0
                penalty += novelty_penalty * common_ratio
            
            # Apply penalty
            adjusted_score = score * (1.0 - penalty)
            penalized_results.append((chunk, adjusted_score))
            
            # Add to selected contents for diversity calculation
            selected_contents.append(chunk.content)
        
        return penalized_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate simple content similarity based on common words.
        
        Args:
            content1: First content
            content2: Second content
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class RetrievalEngine:
    """Main retrieval engine that coordinates different retrieval methods."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        embedding_manager: EmbeddingManager
    ):
        """
        Initialize retrieval engine.
        
        Args:
            vector_store_manager: Vector store manager
            embedding_manager: Embedding manager
        """
        self.vector_store_manager = vector_store_manager
        self.embedding_manager = embedding_manager
        self.retrievers = {}
        self.reranker = None
        
        # Default configuration
        self.default_retriever = "hybrid"
        self.default_k = 5
    
    def add_dense_retriever(
        self,
        name: str,
        vector_store: BaseVectorStore,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """Add a dense retriever."""
        embedding_mgr = embedding_manager or self.embedding_manager
        retriever = DenseRetriever(vector_store, embedding_mgr, name)
        self.retrievers[name] = retriever
        
        logger.info(f"Added dense retriever: {name}")
    
    def add_sparse_retriever(
        self,
        name: str,
        documents: List[DocumentChunk],
        **kwargs
    ):
        """Add a sparse retriever."""
        retriever = SparseRetriever(documents, name, **kwargs)
        self.retrievers[name] = retriever
        
        logger.info(f"Added sparse retriever: {name}")
    
    def add_hybrid_retriever(
        self,
        name: str,
        dense_retriever_name: str,
        sparse_retriever_name: str,
        **kwargs
    ):
        """Add a hybrid retriever."""
        if dense_retriever_name not in self.retrievers:
            raise ValueError(f"Dense retriever {dense_retriever_name} not found")
        
        if sparse_retriever_name not in self.retrievers:
            raise ValueError(f"Sparse retriever {sparse_retriever_name} not found")
        
        dense_retriever = self.retrievers[dense_retriever_name]
        sparse_retriever = self.retrievers[sparse_retriever_name]
        
        if not isinstance(dense_retriever, DenseRetriever):
            raise ValueError(f"{dense_retriever_name} is not a dense retriever")
        
        if not isinstance(sparse_retriever, SparseRetriever):
            raise ValueError(f"{sparse_retriever_name} is not a sparse retriever")
        
        retriever = HybridRetriever(dense_retriever, sparse_retriever, name, **kwargs)
        self.retrievers[name] = retriever
        
        logger.info(f"Added hybrid retriever: {name}")
    
    def set_reranker(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
        """Set up reranker."""
        self.reranker = Reranker(model_name, **kwargs)
        logger.info(f"Set up reranker with model: {model_name}")
    
    async def retrieve(
        self,
        query: str,
        retriever_name: Optional[str] = None,
        k: Optional[int] = None,
        use_reranker: bool = False,
        rerank_top_k: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve documents using specified retriever.
        
        Args:
            query: Query string
            retriever_name: Name of retriever to use
            k: Number of documents to retrieve
            use_reranker: Whether to use reranker
            rerank_top_k: Number of top results after reranking
            **kwargs: Additional parameters
            
        Returns:
            Retrieved and optionally reranked results
        """
        retriever_name = retriever_name or self.default_retriever
        k = k or self.default_k
        
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever {retriever_name} not found")
        
        # Retrieve documents
        retriever = self.retrievers[retriever_name]
        results = await retriever.retrieve(query, k=k, **kwargs)
        
        # Apply reranking if requested
        if use_reranker and self.reranker and results:
            results = await self.reranker.rerank(
                query, results, top_k=rerank_top_k or k, **kwargs
            )
        
        return results
    
    def get_available_retrievers(self) -> List[str]:
        """Get list of available retrievers."""
        return list(self.retrievers.keys())
    
    def get_retriever_info(self, retriever_name: str) -> Dict[str, Any]:
        """Get information about a retriever."""
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever {retriever_name} not found")
        
        retriever = self.retrievers[retriever_name]
        
        info = {
            'name': retriever_name,
            'type': retriever.__class__.__name__,
            'retriever_name': retriever.get_name()
        }
        
        # Add type-specific information
        if isinstance(retriever, HybridRetriever):
            info.update({
                'dense_weight': retriever.dense_weight,
                'sparse_weight': retriever.sparse_weight,
                'dense_retriever': retriever.dense_retriever.get_name(),
                'sparse_retriever': retriever.sparse_retriever.get_name()
            })
        elif isinstance(retriever, SparseRetriever):
            info.update({
                'document_count': len(retriever.documents),
                'vocabulary_size': len(retriever.inverted_index),
                'avg_doc_length': retriever.avg_doc_length,
                'k1': retriever.k1,
                'b': retriever.b
            })
        
        return info
    
    async def evaluate_retrieval(
        self,
        test_queries: List[str],
        ground_truth: List[List[str]],
        retriever_name: Optional[str] = None,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            test_queries: List of test queries
            ground_truth: List of relevant document IDs for each query
            retriever_name: Name of retriever to evaluate
            k: Number of documents to retrieve
            
        Returns:
            Evaluation metrics
        """
        if len(test_queries) != len(ground_truth):
            raise ValueError("Number of queries and ground truth must match")
        
        retriever_name = retriever_name or self.default_retriever
        
        precision_scores = []
        recall_scores = []
        
        for query, relevant_docs in zip(test_queries, ground_truth):
            # Retrieve documents
            results = await self.retrieve(query, retriever_name=retriever_name, k=k)
            retrieved_ids = [chunk.id for chunk, _ in results]
            
            # Calculate precision and recall
            relevant_set = set(relevant_docs)
            retrieved_set = set(retrieved_ids)
            
            if retrieved_set:
                precision = len(relevant_set.intersection(retrieved_set)) / len(retrieved_set)
                precision_scores.append(precision)
            
            if relevant_set:
                recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set)
                recall_scores.append(recall)
        
        # Calculate average metrics
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': f1_score,
            'num_queries': len(test_queries)
        }