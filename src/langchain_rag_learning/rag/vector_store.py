"""Vector storage and retrieval system for document embeddings."""

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from langchain_rag_learning.core.config import get_settings
from langchain_rag_learning.core.exceptions import DocumentProcessingError
from langchain_rag_learning.core.models import DocumentChunk
from langchain_rag_learning.rag.embeddings import batch_cosine_similarity

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Base class for vector storage implementations."""
    
    def __init__(self, collection_name: str, embedding_dimension: int):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection/index
            embedding_dimension: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
    
    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass
    
    @abstractmethod
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector database implementation."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_dimension: int,
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_dimension: Dimension of embedding vectors
            persist_directory: Directory to persist the database
            **kwargs: Additional Chroma configuration
        """
        super().__init__(collection_name, embedding_dimension)
        
        self.persist_directory = persist_directory or get_settings().vector_store_path
        self.client = None
        self.collection = None
        self.config = kwargs
    
    async def _initialize_client(self):
        """Initialize Chroma client and collection."""
        if self.client is not None:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory if it doesn't exist
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize client
            if self.persist_directory:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                self.client = chromadb.Client()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": self.embedding_dimension}
                )
            
            logger.info(f"Initialized Chroma collection: {self.collection_name}")
            
        except ImportError:
            raise DocumentProcessingError(
                "chromadb library not installed. Install with: pip install chromadb"
            )
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to Chroma."""
        await self._initialize_client()
        
        if not chunks:
            return []
        
        try:
            # Prepare data for Chroma
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                if not chunk.embedding:
                    raise DocumentProcessingError(f"Chunk {chunk.id} has no embedding")
                
                ids.append(chunk.id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.content)
                
                # Prepare metadata (Chroma requires string values)
                metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)
                    elif isinstance(value, dict):
                        metadata[key] = json.dumps(value)
                    else:
                        metadata[key] = str(value)
                
                # Add chunk-specific metadata
                metadata.update({
                    "document_id": chunk.document_id,
                    "chunk_index": str(chunk.chunk_index),
                    "start_char": str(chunk.start_char),
                    "end_char": str(chunk.end_char)
                })
                
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to Chroma collection")
            return ids
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to add documents to Chroma: {str(e)}")
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents in Chroma."""
        await self._initialize_client()
        
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filter_dict:
                where_clause = {}
                for key, value in filter_dict.items():
                    where_clause[key] = str(value)
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Convert results to DocumentChunk objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    document = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    embedding = results["embeddings"][0][i] if results["embeddings"] else None
                    
                    # Convert distance to similarity score (Chroma uses L2 distance)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    # Reconstruct DocumentChunk
                    chunk_metadata = {}
                    for key, value in metadata.items():
                        if key in ["document_id", "chunk_index", "start_char", "end_char"]:
                            continue
                        
                        # Try to parse JSON metadata
                        try:
                            chunk_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            chunk_metadata[key] = value
                    
                    document_chunk = DocumentChunk(
                        id=chunk_id,
                        content=document,
                        document_id=metadata.get("document_id", ""),
                        chunk_index=int(metadata.get("chunk_index", 0)),
                        start_char=int(metadata.get("start_char", 0)),
                        end_char=int(metadata.get("end_char", 0)),
                        embedding=embedding,
                        metadata=chunk_metadata
                    )
                    
                    search_results.append((document_chunk, similarity_score))
            
            return search_results
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to search in Chroma: {str(e)}")
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Chroma."""
        await self._initialize_client()
        
        try:
            # Delete by document_id metadata
            for doc_id in document_ids:
                self.collection.delete(where={"document_id": doc_id})
            
            logger.info(f"Deleted documents: {document_ids}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from Chroma: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Chroma collection statistics."""
        await self._initialize_client()
        
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": self.embedding_dimension,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Failed to get Chroma stats: {str(e)}")
            return {}
    
    async def clear_collection(self) -> bool:
        """Clear Chroma collection."""
        await self._initialize_client()
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"dimension": self.embedding_dimension}
            )
            
            logger.info(f"Cleared Chroma collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear Chroma collection: {str(e)}")
            return False


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector database implementation."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_dimension: int,
        index_type: str = "flat",
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_dimension: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            persist_directory: Directory to persist the index
            **kwargs: Additional FAISS configuration
        """
        super().__init__(collection_name, embedding_dimension)
        
        self.index_type = index_type
        self.persist_directory = persist_directory or get_settings().vector_store_path
        self.index = None
        self.id_to_chunk = {}  # Map FAISS IDs to DocumentChunk objects
        self.chunk_id_to_faiss_id = {}  # Map chunk IDs to FAISS IDs
        self.next_id = 0
        self.config = kwargs
    
    async def _initialize_index(self):
        """Initialize FAISS index."""
        if self.index is not None:
            return
        
        try:
            import faiss
            
            # Create index based on type
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product (cosine similarity)
            elif self.index_type == "ivf":
                nlist = self.config.get("nlist", 100)
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
            elif self.index_type == "hnsw":
                m = self.config.get("m", 16)
                self.index = faiss.IndexHNSWFlat(self.embedding_dimension, m)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Try to load existing index
            await self._load_index()
            
            logger.info(f"Initialized FAISS index: {self.index_type}")
            
        except ImportError:
            raise DocumentProcessingError(
                "faiss library not installed. Install with: pip install faiss-cpu or faiss-gpu"
            )
    
    async def _load_index(self):
        """Load existing FAISS index from disk."""
        if not self.persist_directory:
            return
        
        index_path = Path(self.persist_directory) / f"{self.collection_name}.faiss"
        metadata_path = Path(self.persist_directory) / f"{self.collection_name}_metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                import faiss
                
                # Load index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.id_to_chunk = metadata.get('id_to_chunk', {})
                    self.chunk_id_to_faiss_id = metadata.get('chunk_id_to_faiss_id', {})
                    self.next_id = metadata.get('next_id', 0)
                
                logger.info(f"Loaded existing FAISS index with {len(self.id_to_chunk)} documents")
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                # Reset if loading fails
                self.id_to_chunk = {}
                self.chunk_id_to_faiss_id = {}
                self.next_id = 0
    
    async def _save_index(self):
        """Save FAISS index to disk."""
        if not self.persist_directory:
            return
        
        try:
            import faiss
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            index_path = Path(self.persist_directory) / f"{self.collection_name}.faiss"
            metadata_path = Path(self.persist_directory) / f"{self.collection_name}_metadata.pkl"
            
            # Save index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'id_to_chunk': self.id_to_chunk,
                'chunk_id_to_faiss_id': self.chunk_id_to_faiss_id,
                'next_id': self.next_id
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to FAISS."""
        await self._initialize_index()
        
        if not chunks:
            return []
        
        try:
            # Prepare embeddings and metadata
            embeddings = []
            faiss_ids = []
            
            for chunk in chunks:
                if not chunk.embedding:
                    raise DocumentProcessingError(f"Chunk {chunk.id} has no embedding")
                
                # Normalize embedding for cosine similarity
                embedding = np.array(chunk.embedding, dtype=np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                
                embeddings.append(embedding)
                
                # Assign FAISS ID
                faiss_id = self.next_id
                faiss_ids.append(faiss_id)
                
                # Store mappings
                self.id_to_chunk[faiss_id] = chunk
                self.chunk_id_to_faiss_id[chunk.id] = faiss_id
                
                self.next_id += 1
            
            # Add to index
            embeddings_array = np.array(embeddings)
            
            if self.index_type == "ivf" and not self.index.is_trained:
                # Train IVF index if not trained
                if embeddings_array.shape[0] >= self.config.get("nlist", 100):
                    self.index.train(embeddings_array)
                else:
                    logger.warning("Not enough data to train IVF index, using flat index")
            
            self.index.add(embeddings_array)
            
            # Save index
            await self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to FAISS index")
            return [chunk.id for chunk in chunks]
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to add documents to FAISS: {str(e)}")
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents in FAISS."""
        await self._initialize_index()
        
        try:
            # Normalize query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = query_array / np.linalg.norm(query_array)
            
            # Search index
            scores, indices = self.index.search(query_array, k)
            
            # Convert results
            search_results = []
            
            for i in range(len(indices[0])):
                faiss_id = indices[0][i]
                score = float(scores[0][i])
                
                if faiss_id == -1:  # FAISS returns -1 for invalid results
                    continue
                
                if faiss_id in self.id_to_chunk:
                    chunk = self.id_to_chunk[faiss_id]
                    
                    # Apply metadata filtering if specified
                    if filter_dict:
                        match = True
                        for key, value in filter_dict.items():
                            if key not in chunk.metadata or chunk.metadata[key] != value:
                                match = False
                                break
                        
                        if not match:
                            continue
                    
                    search_results.append((chunk, score))
            
            return search_results
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to search in FAISS: {str(e)}")
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS."""
        await self._initialize_index()
        
        try:
            # FAISS doesn't support direct deletion, so we need to rebuild the index
            # Remove chunks from mappings
            faiss_ids_to_remove = []
            
            for doc_id in document_ids:
                # Find all chunks belonging to this document
                chunks_to_remove = []
                for faiss_id, chunk in self.id_to_chunk.items():
                    if chunk.document_id == doc_id:
                        chunks_to_remove.append((faiss_id, chunk.id))
                
                for faiss_id, chunk_id in chunks_to_remove:
                    faiss_ids_to_remove.append(faiss_id)
                    del self.id_to_chunk[faiss_id]
                    if chunk_id in self.chunk_id_to_faiss_id:
                        del self.chunk_id_to_faiss_id[chunk_id]
            
            # If we removed documents, rebuild the index
            if faiss_ids_to_remove:
                await self._rebuild_index()
            
            logger.info(f"Deleted documents: {document_ids}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from FAISS: {str(e)}")
            return False
    
    async def _rebuild_index(self):
        """Rebuild FAISS index after deletions."""
        try:
            import faiss
            
            # Create new index
            if self.index_type == "flat":
                new_index = faiss.IndexFlatIP(self.embedding_dimension)
            elif self.index_type == "ivf":
                nlist = self.config.get("nlist", 100)
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
            elif self.index_type == "hnsw":
                m = self.config.get("m", 16)
                new_index = faiss.IndexHNSWFlat(self.embedding_dimension, m)
            
            # Re-add all remaining chunks
            if self.id_to_chunk:
                embeddings = []
                new_id_to_chunk = {}
                new_chunk_id_to_faiss_id = {}
                new_id = 0
                
                for old_faiss_id, chunk in self.id_to_chunk.items():
                    if chunk.embedding:
                        embedding = np.array(chunk.embedding, dtype=np.float32)
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                        
                        new_id_to_chunk[new_id] = chunk
                        new_chunk_id_to_faiss_id[chunk.id] = new_id
                        new_id += 1
                
                if embeddings:
                    embeddings_array = np.array(embeddings)
                    
                    if self.index_type == "ivf":
                        if embeddings_array.shape[0] >= self.config.get("nlist", 100):
                            new_index.train(embeddings_array)
                    
                    new_index.add(embeddings_array)
                
                # Update mappings
                self.id_to_chunk = new_id_to_chunk
                self.chunk_id_to_faiss_id = new_chunk_id_to_faiss_id
                self.next_id = new_id
            
            # Replace old index
            self.index = new_index
            
            # Save updated index
            await self._save_index()
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get FAISS collection statistics."""
        await self._initialize_index()
        
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.id_to_chunk),
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "persist_directory": self.persist_directory,
            "is_trained": getattr(self.index, 'is_trained', True)
        }
    
    async def clear_collection(self) -> bool:
        """Clear FAISS collection."""
        try:
            # Reset all data structures
            self.id_to_chunk = {}
            self.chunk_id_to_faiss_id = {}
            self.next_id = 0
            
            # Reinitialize index
            self.index = None
            await self._initialize_index()
            
            # Save empty index
            await self._save_index()
            
            logger.info(f"Cleared FAISS collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear FAISS collection: {str(e)}")
            return False


class VectorStoreManager:
    """Manager for different vector store implementations."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self.stores = {}
        self.default_store_type = "chroma"
    
    def create_store(
        self,
        store_type: str,
        collection_name: str,
        embedding_dimension: int,
        **kwargs
    ) -> BaseVectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of vector store ('chroma', 'faiss')
            collection_name: Name of the collection
            embedding_dimension: Dimension of embeddings
            **kwargs: Additional configuration
            
        Returns:
            Vector store instance
        """
        store_key = f"{store_type}_{collection_name}"
        
        if store_key in self.stores:
            return self.stores[store_key]
        
        if store_type == "chroma":
            store = ChromaVectorStore(collection_name, embedding_dimension, **kwargs)
        elif store_type == "faiss":
            store = FAISSVectorStore(collection_name, embedding_dimension, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        self.stores[store_key] = store
        return store
    
    def get_store(self, store_type: str, collection_name: str) -> Optional[BaseVectorStore]:
        """Get existing vector store instance."""
        store_key = f"{store_type}_{collection_name}"
        return self.stores.get(store_key)
    
    def get_available_stores(self) -> List[str]:
        """Get list of available vector store types."""
        return ["chroma", "faiss"]
    
    async def hybrid_search(
        self,
        stores: List[BaseVectorStore],
        query_embedding: List[float],
        k: int = 5,
        weights: Optional[List[float]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform hybrid search across multiple vector stores.
        
        Args:
            stores: List of vector stores to search
            query_embedding: Query embedding vector
            k: Number of results per store
            weights: Weights for combining results from different stores
            filter_dict: Optional metadata filters
            
        Returns:
            Combined and ranked search results
        """
        if not stores:
            return []
        
        weights = weights or [1.0] * len(stores)
        if len(weights) != len(stores):
            raise ValueError("Number of weights must match number of stores")
        
        # Search each store
        all_results = []
        
        for i, store in enumerate(stores):
            try:
                results = await store.similarity_search(
                    query_embedding, k=k, filter_dict=filter_dict
                )
                
                # Apply weight to scores
                weighted_results = [
                    (chunk, score * weights[i]) for chunk, score in results
                ]
                
                all_results.extend(weighted_results)
                
            except Exception as e:
                logger.warning(f"Search failed for store {store.collection_name}: {e}")
                continue
        
        # Combine and deduplicate results
        chunk_scores = {}
        
        for chunk, score in all_results:
            chunk_id = chunk.id
            if chunk_id in chunk_scores:
                # Take the maximum score for duplicates
                chunk_scores[chunk_id] = (chunk, max(chunk_scores[chunk_id][1], score))
            else:
                chunk_scores[chunk_id] = (chunk, score)
        
        # Sort by score and return top k
        final_results = list(chunk_scores.values())
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:k]