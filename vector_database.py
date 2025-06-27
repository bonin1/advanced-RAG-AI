import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Abstract base class for vector databases."""
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add documents with embeddings to the database."""
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, k: int = config.MAX_CHUNKS_CONTEXT) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """Save the database."""
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """Load the database."""
        raise NotImplementedError

class FAISSDatabase(VectorDatabase):
    """FAISS-based vector database for local storage."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        self.documents = []
        self.doc_ids = []
        
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add documents with embeddings to FAISS index."""
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store documents and IDs
            start_id = len(self.documents)
            for i, chunk in enumerate(chunks):
                chunk['doc_id'] = start_id + i
                self.documents.append(chunk)
                self.doc_ids.append(start_id + i)
            
            logger.info(f"Added {len(chunks)} documents to FAISS index")
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = config.MAX_CHUNKS_CONTEXT) -> List[Dict[str, Any]]:
        """Search for similar documents in FAISS index with better similarity calculation."""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            # Return results with scores
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    # Convert inner product back to meaningful similarity score
                    doc['similarity_score'] = max(0.0, float(score))  # Ensure non-negative
                    doc['rank'] = i + 1
                    results.append(doc)
            
            # Sort by similarity score (higher is better)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []
    
    def save(self, path: str) -> None:
        """Save FAISS index and documents."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(path, "faiss_index.idx"))
            
            # Save documents
            with open(os.path.join(path, "documents.pkl"), 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save metadata
            metadata = {
                'dimension': self.dimension,
                'total_docs': len(self.documents)
            }
            with open(os.path.join(path, "metadata.pkl"), 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"FAISS database saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS database: {e}")
            raise
    
    def load(self, path: str) -> None:
        """Load FAISS index and documents."""
        try:
            index_path = os.path.join(path, "faiss_index.idx")
            docs_path = os.path.join(path, "documents.pkl")
            meta_path = os.path.join(path, "metadata.pkl")
            
            if not all(os.path.exists(p) for p in [index_path, docs_path, meta_path]):
                logger.warning(f"FAISS database not found at {path}")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load metadata
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
                self.dimension = metadata['dimension']
            
            # Rebuild doc_ids
            self.doc_ids = list(range(len(self.documents)))
            
            logger.info(f"FAISS database loaded from {path}, {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading FAISS database: {e}")
            raise

class ChromaDatabase(VectorDatabase):
    """ChromaDB-based vector database."""
    
    def __init__(self, collection_name: str = "rag_documents"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
    def _initialize_client(self, path: str) -> None:
        """Initialize ChromaDB client."""
        try:
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add documents with embeddings to ChromaDB."""
        try:
            if self.collection is None:
                raise ValueError("ChromaDB not initialized")
            
            # Prepare data for ChromaDB
            ids = [f"doc_{i}_{chunk.get('chunk_id', 0)}" for i, chunk in enumerate(chunks)]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk.get('metadata', {}) for chunk in chunks]
            embeddings_list = embeddings.tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = config.MAX_CHUNKS_CONTEXT) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB."""
        try:
            if self.collection is None:
                return []
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    result = {
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1.0 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def save(self, path: str) -> None:
        """ChromaDB automatically persists data."""
        logger.info(f"ChromaDB data persisted to {path}")
    
    def load(self, path: str) -> None:
        """Load ChromaDB from path."""
        self._initialize_client(path)

class VectorDatabaseManager:
    """Manager for vector database operations."""
    
    def __init__(self, db_type: str = config.VECTOR_DB_TYPE):
        self.db_type = db_type
        self.database = None
        self.db_path = Path(config.VECTOR_DB_PATH)
        
        # Initialize database
        if db_type == "faiss":
            self.database = FAISSDatabase()
        elif db_type == "chroma":
            self.database = ChromaDatabase()
            self.database._initialize_client(str(self.db_path))
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add documents to the vector database."""
        self.database.add_documents(chunks, embeddings)
    
    def search_similar(self, query_embedding: np.ndarray, k: int = config.MAX_CHUNKS_CONTEXT) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        return self.database.search(query_embedding, k)
    
    def save_database(self) -> None:
        """Save the database."""
        self.database.save(str(self.db_path))
    
    def load_database(self) -> None:
        """Load the database."""
        self.database.load(str(self.db_path))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.db_type == "faiss" and hasattr(self.database, 'index'):
            return {
                'type': 'FAISS',
                'total_documents': len(self.database.documents),
                'index_size': self.database.index.ntotal,
                'dimension': self.database.dimension
            }
        elif self.db_type == "chroma" and self.database.collection:
            count = self.database.collection.count()
            return {
                'type': 'ChromaDB',
                'total_documents': count,
                'collection_name': self.database.collection_name
            }
        else:
            return {'type': self.db_type, 'status': 'not_initialized'}
