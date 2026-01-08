"""Retrieval engines for RAG system including dense, sparse, and hybrid retrievers."""

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from langchain_rag_learning.core.config import get_settings
from langchain_rag_learning.core.exceptions import DocumentProcessingError
from langchain_rag_learning.core.models import DocumentChunk
from langchain_rag_learning.rag.embeddings import EmbeddingManager, cosine_similarity
from langchain_rag_learning.rag.vector_store import BaseVectorStore, VectorStoreManager

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Base class for all retrievers."""
    
    def __init__(self, name: str):
        """
        Initialize base retriever.
        
        Args:
            name: Name of the retriever
        """
        self.name = name
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            **kwargs: Additional retriever-specific parameters
            
        Returns:
            List of (DocumentChunk, relevance_score) tuples
        """
        pass
    
    def get_name(self) -> str:
        """Get retriever name."""
        return self.name


class DenseRetriever(BaseRetriever):
    """Dense retriever using vector similarity search."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_manager: EmbeddingManager,
        name: str = "dense_retriever"
    ):
        """
        Initialize dense retriever.
        
        Args:
            vector_store: Vector store for similarity search
            embedding_manager: Embedding manager for query embedding
            name: Name of the retriever
        """
        super().__init__(name)
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
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
        Retrieve documents using dense vector similarity.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            embedding_provider: Embedding provider to use
            similarity_threshold: Minimum similarity threshold
            **kwargs: Additional parameters
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        try:
            start_time = time.time()
            
            # Embed the query
            query_embedding = await self.embedding_manager.embed_query(
                query, provider=embedding_provider
            )
            
            # Perform similarity search
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=k,
                filter_dict=filter_dict
            )
            
            # Filter by similarity threshold
            if similarity_threshold > 0:
                results = [
                    (chunk, score) for chunk, score in results
                    if score >= similarity_threshold
                ]
            
            # Add retrieval metadata
            retrieval_time = time.time() - start_time
            
            for chunk, score in results:
                if 'retrieval_info' not in chunk.metadata:
                    chunk.metadata['retrieval_info'] = {}
                
                chunk.metadata['retrieval_info'].update({
                    'retriever': self.name,
                    'similarity_score': score,
                    'retrieval_time': retrieval_time,
                    'query': query[:100]  # Store truncated query
                })
            
            logger.info(f"Dense retrieval found {len(results)} documents in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            raise DocumentProcessingError(f"Dense retrieval failed: {str(e)}")
    
    async def get_query_expansion_terms(
        self,
        query: str,
        expansion_count: int = 5
    ) -> List[str]:
        """
        Get query expansion terms using semantic similarity.
        
        Args:
            query: Original query
            expansion_count: Number of expansion terms to generate
            
        Returns:
            List of expansion terms
        """
        try:
            # Get initial results
            initial_results = await self.retrieve(query, k=10)
            
            if not initial_results:
                return []
            
            # Extract key terms from top results
            all_text = " ".join([chunk.content for chunk, _ in initial_results[:3]])
            
            # Simple term extraction (can be enhanced with NLP)
            words = all_text.lower().split()
            word_freq = Counter(words)
            
            # Filter out common words and get most frequent terms
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            expansion_terms = [
                word for word, freq in word_freq.most_common(expansion_count * 2)
                if word not in common_words and len(word) > 2 and word not in query.lower()
            ]
            
            return expansion_terms[:expansion_count]
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []


class SparseRetriever(BaseRetriever):
    """Sparse retriever using keyword-based search (BM25 and TF-IDF)."""
    
    def __init__(
        self,
        documents: List[DocumentChunk],
        name: str = "sparse_retriever",
        k1: float = 1.2,
        b: float = 0.75
    ):
        """
        Initialize sparse retriever.
        
        Args:
            documents: List of document chunks to index
            name: Name of the retriever
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        super().__init__(name)
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        # Build inverted index
        self.inverted_index = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.doc_frequencies = {}
        self.total_docs = len(documents)
        
        self._build_index()
    
    def _build_index(self):
        """Build inverted index for BM25 scoring."""
        logger.info(f"Building sparse index for {len(self.documents)} documents")
        
        # Tokenize documents and build inverted index
        total_length = 0
        
        for i, doc in enumerate(self.documents):
            # Simple tokenization (can be enhanced with proper NLP)
            tokens = self._tokenize(doc.content)
            doc_length = len(tokens)
            
            self.doc_lengths[i] = doc_length
            total_length += doc_length
            
            # Count term frequencies in document
            term_freq = Counter(tokens)
            
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                
                self.inverted_index[term][i] = freq
        
        # Calculate average document length
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
        
        # Calculate document frequencies
        for term, doc_dict in self.inverted_index.items():
            self.doc_frequencies[term] = len(doc_dict)
        
        logger.info(f"Built index with {len(self.inverted_index)} unique terms")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on whitespace and punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if len(token) > 2]  # Filter short tokens
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_index: int) -> float:
        """
        Calculate BM25 score for a document.
        
        Args:
            query_terms: List of query terms
            doc_index: Document index
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.doc_lengths.get(doc_index, 0)
        
        for term in query_terms:
            if term in self.inverted_index and doc_index in self.inverted_index[term]:
                # Term frequency in document
                tf = self.inverted_index[term][doc_index]
                
                # Document frequency
                df = self.doc_frequencies[term]
                
                # IDF calculation
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def _calculate_tfidf_score(self, query_terms: List[str], doc_index: int) -> float:
        """
        Calculate TF-IDF score for a document.
        
        Args:
            query_terms: List of query terms
            doc_index: Document index
            
        Returns:
            TF-IDF score
        """
        score = 0.0
        doc_length = self.doc_lengths.get(doc_index, 0)
        
        for term in query_terms:
            if term in self.inverted_index and doc_index in self.inverted_index[term]:
                # Term frequency
                tf = self.inverted_index[term][doc_index] / doc_length if doc_length > 0 else 0
                
                # Inverse document frequency
                df = self.doc_frequencies[term]
                idf = math.log(self.total_docs / df) if df > 0 else 0
                
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
        Retrieve documents using sparse retrieval.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            scoring_method: Scoring method ('bm25' or 'tfidf')
            query_expansion: Whether to perform query expansion
            **kwargs: Additional parameters
            
        Returns:
            List of (DocumentChunk, relevance_score) tuples
        """
        try:
            start_time = time.time()
            
            # Tokenize query
            query_terms = self._tokenize(query)
            
            if not query_terms:
                return []
            
            # Perform query expansion if requested
            if query_expansion:
                expanded_terms = await self._expand_query(query_terms)
                query_terms.extend(expanded_terms)
                query_terms = list(set(query_terms))  # Remove duplicates
            
            # Calculate scores for all documents
            doc_scores = []
            
            for i, doc in enumerate(self.documents):
                # Apply metadata filters
                if filter_dict:
                    match = True
                    for key, value in filter_dict.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                # Calculate score based on method
                if scoring_method == "bm25":
                    score = self._calculate_bm25_score(query_terms, i)
                elif scoring_method == "tfidf":
                    score = self._calculate_tfidf_score(query_terms, i)
                else:
                    raise ValueError(f"Unknown scoring method: {scoring_method}")
                
                if score > 0:
                    doc_scores.append((doc, score))
            
            # Sort by score and take top k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            results = doc_scores[:k]
            
            # Add retrieval metadata
            retrieval_time = time.time() - start_time
            
            for chunk, score in results:
                if 'retrieval_info' not in chunk.metadata:
                    chunk.metadata['retrieval_info'] = {}
                
                chunk.metadata['retrieval_info'].update({
                    'retriever': self.name,
                    'relevance_score': score,
                    'scoring_method': scoring_method,
                    'retrieval_time': retrieval_time,
                    'query_terms': query_terms,
                    'query': query[:100]
                })
            
            logger.info(f"Sparse retrieval found {len(results)} documents in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            raise DocumentProcessingError(f"Sparse retrieval failed: {str(e)}")
    
    async def _expand_query(self, query_terms: List[str]) -> List[str]:
        """
        Expand query with related terms.
        
        Args:
            query_terms: Original query terms
            
        Returns:
            List of expansion terms
        """
        expansion_terms = []
        
        # Simple co-occurrence based expansion
        term_cooccurrence = defaultdict(int)
        
        for term in query_terms:
            if term in self.inverted_index:
                # Find documents containing this term
                docs_with_term = self.inverted_index[term].keys()
                
                # Find other terms in these documents
                for doc_idx in docs_with_term:
                    doc_tokens = self._tokenize(self.documents[doc_idx].content)
                    for token in doc_tokens:
                        if token != term and token not in query_terms:
                            term_cooccurrence[token] += 1
        
        # Get top co-occurring terms
        expansion_terms = [
            term for term, count in sorted(term_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        
        return expansion_terms
    
    def update_index(self, new_documents: List[DocumentChunk]):
        """
        Update the index with new documents.
        
        Args:
            new_documents: New documents to add to the index
        """
        self.documents.extend(new_documents)
        self.total_docs = len(self.documents)
        self._build_index()
        
        logger.info(f"Updated sparse index with {len(new_documents)} new documents")


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and sparse retrieval methods."""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        name: str = "hybrid_retriever",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Sparse retriever instance
            name: Name of the retriever
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
        """
        super().__init__(name)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Validate weights
        if abs(dense_weight + sparse_weight - 1.0) > 1e-6:
            logger.warning(f"Weights don't sum to 1.0: {dense_weight} + {sparse_weight} = {dense_weight + sparse_weight}")
    
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
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            fusion_method: Fusion method ('rrf', 'weighted', 'max')
            rrf_k: RRF parameter k
            normalize_scores: Whether to normalize scores before fusion
            **kwargs: Additional parameters
            
        Returns:
            List of (DocumentChunk, fused_score) tuples
        """
        try:
            start_time = time.time()
            
            # Retrieve from both retrievers in parallel
            dense_task = asyncio.create_task(
                self.dense_retriever.retrieve(query, k=k*2, filter_dict=filter_dict, **kwargs)
            )
            sparse_task = asyncio.create_task(
                self.sparse_retriever.retrieve(query, k=k*2, filter_dict=filter_dict, **kwargs)
            )
            
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            
            # Fuse results based on method
            if fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, rrf_k)
            elif fusion_method == "weighted":
                fused_results = self._weighted_fusion(dense_results, sparse_results, normalize_scores)
            elif fusion_method == "max":
                fused_results = self._max_fusion(dense_results, sparse_results, normalize_scores)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            # Take top k results
            final_results = fused_results[:k]
            
            # Add retrieval metadata
            retrieval_time = time.time() - start_time
            
            for chunk, score in final_results:
                if 'retrieval_info' not in chunk.metadata:
                    chunk.metadata['retrieval_info'] = {}
                
                chunk.metadata['retrieval_info'].update({
                    'retriever': self.name,
                    'fusion_method': fusion_method,
                    'hybrid_score': score,
                    'retrieval_time': retrieval_time,
                    'dense_weight': self.dense_weight,
                    'sparse_weight': self.sparse_weight,
                    'query': query[:100]
                })
            
            logger.info(f"Hybrid retrieval found {len(final_results)} documents in {retrieval_time:.3f}s")
            return final_results
            
        except Exception as e:
            raise DocumentProcessingError(f"Hybrid retrieval failed: {str(e)}")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        k: int = 60
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform Reciprocal Rank Fusion (RRF).
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            k: RRF parameter
            
        Returns:
            Fused results sorted by RRF score
        """
        # Create rank mappings
        dense_ranks = {chunk.id: rank + 1 for rank, (chunk, _) in enumerate(dense_results)}
        sparse_ranks = {chunk.id: rank + 1 for rank, (chunk, _) in enumerate(sparse_results)}
        
        # Collect all unique chunks
        all_chunks = {}
        for chunk, _ in dense_results + sparse_results:
            all_chunks[chunk.id] = chunk
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id, chunk in all_chunks.items():
            rrf_score = 0.0
            
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[chunk_id])
            
            if chunk_id in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[chunk_id])
            
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by RRF score
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
            from sentence_transformers import CrossEncoder
            
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