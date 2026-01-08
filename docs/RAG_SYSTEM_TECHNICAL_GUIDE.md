# RAG System Technical Guide: Advanced Implementation

## Overview
This document explains the technical implementation of the Retrieval-Augmented Generation (RAG) system, covering vector databases, embedding models, retrieval strategies, and advanced NLP concepts.

## Core RAG Architecture

### 1. Document Processing Pipeline
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
import tiktoken

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # Hierarchical splitting
            length_function=len,  # Character-based counting
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
    
    async def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process document through complete pipeline."""
        # 1. Load document
        loader = self._get_loader(file_path)
        documents = loader.load()
        
        # 2. Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # 3. Create embeddings
        embeddings = await self._create_embeddings(chunks)
        
        # 4. Return processed chunks
        return [
            DocumentChunk(
                content=chunk.page_content,
                metadata=chunk.metadata,
                embedding=embedding,
                token_count=len(self.tokenizer.encode(chunk.page_content))
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
```
- **Text Splitting**: Hierarchical document segmentation
- **Overlap Strategy**: Maintains context between chunks
- **Token Counting**: Accurate token usage tracking
- **Metadata Preservation**: Maintains document source information

### 2. Vector Embeddings and Similarity Search
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import faiss

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # FAISS index for efficient similarity search
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        self.document_chunks = []
    
    async def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks."""
        # Batch processing for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,           # Process in batches
            show_progress_bar=True,  # Progress tracking
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        return embeddings
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to vector index."""
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store document references
        self.document_chunks.extend(chunks)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Perform similarity search using vector index."""
        # Create query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        # Return chunks with similarity scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                chunk = self.document_chunks[idx]
                results.append((chunk, float(score)))
        
        return results
```
- **Sentence Transformers**: Pre-trained embedding models
- **FAISS Index**: Efficient similarity search at scale
- **Batch Processing**: Optimized embedding generation
- **Cosine Similarity**: Normalized vector comparison

### 3. Hybrid Retrieval Strategy
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.bm25 = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,      # Vocabulary size limit
            stop_words='english',    # Remove common words
            ngram_range=(1, 2),      # Unigrams and bigrams
            min_df=2,                # Minimum document frequency
            max_df=0.95              # Maximum document frequency
        )
        self.documents = []
    
    def build_sparse_index(self, documents: List[str]):
        """Build BM25 and TF-IDF indices for sparse retrieval."""
        self.documents = documents
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
    
    async def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[DocumentChunk]:
        """Combine dense and sparse retrieval with weighted fusion."""
        # Dense retrieval (semantic similarity)
        dense_results = self.embedding_manager.similarity_search(query, k * 2)
        
        # Sparse retrieval (keyword matching)
        sparse_scores = self.bm25.get_scores(query.lower().split())
        sparse_indices = np.argsort(sparse_scores)[::-1][:k * 2]
        
        # Score normalization
        dense_scores = {idx: score for idx, (chunk, score) in enumerate(dense_results)}
        sparse_scores_norm = self._normalize_scores(sparse_scores)
        
        # Weighted combination (Reciprocal Rank Fusion)
        combined_scores = {}
        
        # Add dense scores
        for rank, (chunk, score) in enumerate(dense_results):
            chunk_idx = self._get_chunk_index(chunk)
            combined_scores[chunk_idx] = alpha * (1.0 / (rank + 1))
        
        # Add sparse scores
        for rank, idx in enumerate(sparse_indices):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * (1.0 / (rank + 1))
            else:
                combined_scores[idx] = (1 - alpha) * (1.0 / (rank + 1))
        
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.embedding_manager.document_chunks[idx] for idx, _ in sorted_results[:k]]
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization of scores."""
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
```
- **Hybrid Approach**: Combines semantic and lexical search
- **BM25 Algorithm**: Probabilistic ranking function
- **TF-IDF**: Term frequency-inverse document frequency
- **Reciprocal Rank Fusion**: Score combination strategy
- **Score Normalization**: Ensures fair comparison between methods

### 4. Advanced Reranking
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
    
    async def rerank(self, query: str, documents: List[DocumentChunk], top_k: int = 5) -> List[DocumentChunk]:
        """Rerank documents using cross-encoder model."""
        if len(documents) <= top_k:
            return documents
        
        # Prepare query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Batch processing for efficiency
        batch_size = 16
        scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = await self._score_batch(batch_pairs)
            scores.extend(batch_scores)
        
        # Sort by relevance score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores[:top_k]]
    
    async def _score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score a batch of query-document pairs."""
        # Tokenize pairs
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]  # Relevance scores
        
        return scores.tolist()
```
- **Cross-Encoder**: Deep interaction between query and document
- **Transformer Architecture**: Attention mechanism for relevance scoring
- **Batch Processing**: Efficient GPU utilization
- **Softmax Normalization**: Probability distribution over relevance

### 5. Query Enhancement and Expansion
```python
import spacy
from collections import Counter
import asyncio

class QueryEnhancer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.synonym_cache = {}
    
    async def enhance_query(self, query: str) -> str:
        """Enhance query with synonyms and related terms."""
        doc = self.nlp(query)
        
        # Extract key entities and concepts
        entities = [ent.text for ent in doc.ents]
        key_terms = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        
        # Generate synonyms
        enhanced_terms = []
        for term in key_terms:
            synonyms = await self._get_synonyms(term)
            enhanced_terms.extend(synonyms[:2])  # Add top 2 synonyms
        
        # Combine original query with enhancements
        enhanced_query = f"{query} {' '.join(enhanced_terms)}"
        return enhanced_query
    
    async def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms using WordNet or external API."""
        if word in self.synonym_cache:
            return self.synonym_cache[word]
        
        # Use NLTK WordNet for synonyms
        from nltk.corpus import wordnet
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        
        synonym_list = list(synonyms)[:5]  # Limit to top 5
        self.synonym_cache[word] = synonym_list
        return synonym_list
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract query intent and entities."""
        doc = self.nlp(query)
        
        # Intent classification based on patterns
        intent = "general"
        if any(word in query.lower() for word in ["how", "what", "why", "when", "where"]):
            intent = "question"
        elif any(word in query.lower() for word in ["find", "search", "show", "list"]):
            intent = "search"
        elif any(word in query.lower() for word in ["explain", "describe", "tell me about"]):
            intent = "explanation"
        
        return {
            "intent": intent,
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "key_terms": [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB"]],
            "sentiment": "neutral"  # Could be enhanced with sentiment analysis
        }
```
- **NLP Pipeline**: spaCy for linguistic analysis
- **Named Entity Recognition**: Extract important entities
- **Query Expansion**: Add synonyms and related terms
- **Intent Classification**: Understand query purpose
- **Lemmatization**: Reduce words to base forms

### 6. Context-Aware Response Generation
```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ContextAwareGenerator:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query", "chat_history"],
            template="""
            You are a helpful AI assistant. Use the following context to answer the user's question.
            
            Context:
            {context}
            
            Chat History:
            {chat_history}
            
            User Question: {query}
            
            Instructions:
            1. Base your answer primarily on the provided context
            2. If the context doesn't contain enough information, say so clearly
            3. Consider the chat history for continuity
            4. Provide specific references to the source material when possible
            5. Be concise but comprehensive
            
            Answer:
            """
        )
    
    async def generate_response(
        self, 
        query: str, 
        context_chunks: List[DocumentChunk],
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate contextually aware response."""
        
        # Prepare context from retrieved chunks
        context = self._format_context(context_chunks)
        
        # Format chat history
        history_text = self._format_chat_history(chat_history or [])
        
        # Generate response
        prompt = self.prompt_template.format(
            context=context,
            query=query,
            chat_history=history_text
        )
        
        response = await self.llm_manager.generate(prompt)
        
        # Extract citations and confidence
        citations = self._extract_citations(response.content, context_chunks)
        confidence = self._calculate_confidence(response, context_chunks)
        
        return {
            "answer": response.content,
            "sources": [chunk.metadata for chunk in context_chunks],
            "citations": citations,
            "confidence": confidence,
            "context_used": len(context_chunks)
        }
    
    def _format_context(self, chunks: List[DocumentChunk]) -> str:
        """Format context chunks for prompt."""
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            formatted_chunks.append(f"[{i}] Source: {source}\n{chunk.content}\n")
        return "\n".join(formatted_chunks)
    
    def _calculate_confidence(self, response, context_chunks: List[DocumentChunk]) -> float:
        """Calculate response confidence based on context overlap."""
        response_words = set(response.content.lower().split())
        context_words = set()
        
        for chunk in context_chunks:
            context_words.update(chunk.content.lower().split())
        
        # Calculate overlap ratio
        overlap = len(response_words.intersection(context_words))
        total_response_words = len(response_words)
        
        if total_response_words == 0:
            return 0.0
        
        confidence = min(overlap / total_response_words, 1.0)
        return confidence
```
- **Prompt Engineering**: Structured prompts for better responses
- **Context Formatting**: Organize retrieved information
- **Citation Extraction**: Link responses to sources
- **Confidence Scoring**: Measure response reliability
- **Chat History**: Maintain conversation context

This technical guide covers the advanced implementation details of the RAG system, explaining both the algorithms and the practical implementation considerations for building a production-ready retrieval-augmented generation system.