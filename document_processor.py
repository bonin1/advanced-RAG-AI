import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document
import config

# Simple text splitter implementation
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, length_function=len, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive character splitting."""
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        # Try each separator
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    # If adding this split would exceed chunk size
                    if self.length_function(current_chunk + separator + split) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Handle overlap
                            overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                            current_chunk = overlap_text + separator + split if overlap_text else split
                        else:
                            current_chunk = split
                    else:
                        current_chunk = current_chunk + separator + split if current_chunk else split
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks
        
        # If no separators work, split by character count
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        
        return chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document parsing, chunking, and preprocessing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create upload directory
        Path(config.UPLOAD_DIR).mkdir(exist_ok=True)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            raise
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from supported file formats."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        try:
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                chunk_obj = {
                    'content': chunk,
                    'chunk_id': i,
                    'metadata': metadata or {}
                }
                chunk_objects.append(chunk_obj)
            
            return chunk_objects
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def process_document(self, file_path: str, document_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a document: extract text and create chunks."""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Extract text
            text = self.extract_text_from_file(file_path)
            
            if not text.strip():
                raise ValueError("No text content found in the document")
            
            # Create metadata
            metadata = {
                'source': document_name or Path(file_path).name,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'text_length': len(text)
            }
            
            # Create chunks
            chunks = self.chunk_text(text, metadata)
            
            logger.info(f"Successfully processed document: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents and return all chunks."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        logger.info(f"Processed {len(file_paths)} documents, created {len(all_chunks)} total chunks")
        return all_chunks
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file is supported and readable."""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in config.ALLOWED_EXTENSIONS:
                return False
            
            # Try to extract a small amount of text to verify readability
            text = self.extract_text_from_file(file_path)
            return len(text.strip()) > 0
            
        except Exception:
            return False
