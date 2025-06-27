import logging
from typing import List, Dict, Any, Optional
import numpy as np
from document_processor import DocumentProcessor
from llm_manager import LLMManager
from vector_database import VectorDatabaseManager
import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system that orchestrates document processing, vector search, and LLM responses."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.llm_manager = LLMManager()
        self.vector_db = VectorDatabaseManager()
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the RAG system by loading models and database."""
        try:
            logger.info("Initializing RAG system with Mistral-7B-Instruct...")
            
            # Load LLM and embedding models
            self.llm_manager.load_models()
            
            # Try to load existing vector database
            try:
                self.vector_db.load_database()
                logger.info("Loaded existing vector database")
            except Exception as e:
                logger.info(f"No existing database found or error loading: {e}")
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully with Mistral-7B-Instruct!")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process and add documents to the vector database."""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        try:
            logger.info(f"Adding {len(file_paths)} documents to the system...")
            
            # Process documents
            chunks = self.document_processor.process_multiple_documents(file_paths)
            
            if not chunks:
                return {"success": False, "message": "No valid documents could be processed"}
            
            # Generate embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.llm_manager.get_embeddings(texts)
            
            # Convert to numpy array if needed
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu().numpy()
            
            # Add to vector database
            self.vector_db.add_documents(chunks, embeddings)
            
            # Save database
            self.vector_db.save_database()
            
            # Return statistics
            stats = self.vector_db.get_database_stats()
            
            return {
                "success": True,
                "chunks_added": len(chunks),
                "files_processed": len(file_paths),
                "database_stats": stats,
                "message": f"Successfully added {len(chunks)} chunks from {len(file_paths)} documents"
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Generate embedding for the question
            question_embedding = self.llm_manager.get_embeddings([question])
            
            # Convert to numpy array if needed
            if hasattr(question_embedding, 'cpu'):
                question_embedding = question_embedding.cpu().numpy()
            
            # Search for relevant chunks
            relevant_chunks = self.vector_db.search_similar(
                question_embedding[0], 
                k=config.MAX_CHUNKS_CONTEXT
            )
            
            if not relevant_chunks:
                return {
                    "success": False,
                    "answer": "I don't have any relevant information to answer your question. Please upload some documents first.",
                    "sources": [],
                    "context_used": []
                }
            
            # Extract content from chunks
            context_texts = [chunk['content'] for chunk in relevant_chunks]
            
            # Create RAG prompt
            rag_prompt = self.llm_manager.create_rag_prompt(question, context_texts)
            
            # Generate response
            answer = self.llm_manager.generate_response(rag_prompt)
            
            # Prepare sources information
            sources = []
            for chunk in relevant_chunks:
                source_info = {
                    "source": chunk.get('metadata', {}).get('source', 'Unknown'),
                    "similarity_score": chunk.get('similarity_score', 0.0),
                    "chunk_preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                }
                sources.append(source_info)
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "context_used": context_texts,
                "num_chunks_used": len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "answer": "I apologize, but I encountered an error while processing your question.",
                "sources": [],
                "context_used": []
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        try:
            db_stats = self.vector_db.get_database_stats()
            
            return {
                "status": "initialized",
                "model_name": self.llm_manager.model_name,
                "embedding_model": config.EMBEDDING_MODEL,
                "device": self.llm_manager.device,
                "database": db_stats,
                "config": {
                    "chunk_size": config.CHUNK_SIZE,
                    "chunk_overlap": config.CHUNK_OVERLAP,
                    "max_chunks_context": config.MAX_CHUNKS_CONTEXT,
                    "max_new_tokens": config.MAX_NEW_TOKENS,
                    "temperature": config.TEMPERATURE
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def clear_database(self) -> Dict[str, Any]:
        """Clear the vector database."""
        try:
            # Reinitialize the database (clears existing data)
            self.vector_db = VectorDatabaseManager()
            
            return {"success": True, "message": "Database cleared successfully"}
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
