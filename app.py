import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List
import time
import traceback

# Import our RAG components
from rag_system import RAGSystem
import config

# Page configuration
st.set_page_config(
    page_title="Advanced RAG AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .source-box {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_system():
    """Initialize the RAG system."""
    try:
        with st.spinner("🤖 Initializing RAG system... This may take a few minutes on first run."):
            st.session_state.rag_system = RAGSystem()
            st.session_state.rag_system.initialize()
            st.session_state.system_initialized = True
        
        # Show which mode we're running in
        if hasattr(st.session_state.rag_system, 'using_full_llm'):
            if st.session_state.rag_system.using_full_llm:
                st.success("✅ RAG system initialized successfully with full Mistral-7B-Instruct!")
            else:
                st.warning("⚠️ RAG system running in simplified mode (TF-IDF + rules)")
                st.info("For full AI capabilities, run: pip install torch transformers sentence-transformers")
        else:
            st.success("✅ RAG system initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.error("Please check your system requirements and try again.")
        with st.expander("Show error details"):
            st.text(str(e))
        return False

def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files and return their paths."""
    saved_paths = []
    
    # Create temporary directory for uploaded files
    temp_dir = Path(config.UPLOAD_DIR)
    temp_dir.mkdir(exist_ok=True)
    
    for uploaded_file in uploaded_files:
        # Save file
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))
    
    return saved_paths

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Advanced RAG AI System</h1>
        <p>Upload documents and ask questions powered by Mistral-7B-Instruct</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Initialize system button
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize RAG System", type="primary"):
                initialize_system()
        else:
            st.success("✅ System Ready")
            
            # System stats
            if st.session_state.rag_system:
                stats = st.session_state.rag_system.get_system_stats()
                st.subheader("Database Stats")
                if 'database' in stats:
                    db_stats = stats['database']
                    st.metric("Documents", db_stats.get('total_documents', 0))
                    st.metric("DB Type", db_stats.get('type', 'Unknown'))
                
                st.subheader("Model Info")
                st.text(f"Device: {stats.get('device', 'Unknown')}")
                st.text(f"Model: Mistral-7B-Instruct")
        
        # Clear database button
        if st.session_state.system_initialized:
            if st.button("🗑️ Clear Database", type="secondary"):
                if st.session_state.rag_system:
                    result = st.session_state.rag_system.clear_database()
                    if result['success']:
                        st.success("Database cleared!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['message']}")
    
    # Main content
    if not st.session_state.system_initialized:
        st.info("👆 Please initialize the RAG system using the sidebar button to get started.")
        
        # System requirements info
        st.subheader("📋 System Requirements")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Hardware Requirements:**
            - 8GB+ RAM (16GB+ recommended)
            - GPU with 8GB+ VRAM (optional but recommended)
            - 20GB+ free disk space
            
            **Supported File Formats:**
            - PDF (.pdf)
            - Word Documents (.docx)
            - Text Files (.txt)
            """)
        
        with col2:
            st.markdown("""
            **Features:**
            - Local LLM (Mistral-7B-Instruct)
            - Vector similarity search (FAISS/ChromaDB)
            - Document chunking and embedding
            - Source attribution
            - No data leaves your computer
            """)
        
        return
    
    # Document upload section
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload documents that the AI will use to answer your questions"
    )
    
    if uploaded_files:
        if st.button("📚 Process Documents", type="primary"):
            try:
                with st.spinner("Processing documents..."):
                    # Save uploaded files
                    file_paths = save_uploaded_files(uploaded_files)
                    
                    # Add to RAG system
                    result = st.session_state.rag_system.add_documents(file_paths)
                
                if result['success']:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Documents processed successfully!</h4>
                        <p>• Files processed: {result['files_processed']}</p>
                        <p>• Chunks created: {result['chunks_added']}</p>
                        <p>• Total documents in database: {result['database_stats'].get('total_documents', 0)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <h4>❌ Error processing documents</h4>
                        <p>{result['message']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.text(traceback.format_exc())
    
    # Question answering section
    st.header("❓ Ask Questions")
    
    # Chat interface
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main topic discussed in the documents?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    
    if ask_button and question:
        if not st.session_state.rag_system:
            st.error("Please initialize the system first.")
            return
        
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.rag_system.query(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'result': result,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                result = None
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        
        # Show most recent first
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:100]}..." if len(chat['question']) > 100 else f"Q: {chat['question']}", expanded=(i == 0)):
                result = chat['result']
                
                if result and result['success']:
                    # Answer
                    st.markdown(f"**🤖 Answer:**")
                    st.markdown(result['answer'])
                    
                    # Sources
                    if result.get('sources'):
                        st.markdown("**📚 Sources:**")
                        for idx, source in enumerate(result['sources']):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {idx + 1}:</strong> {source['source']}<br>
                                <strong>Similarity:</strong> {source['similarity_score']:.3f}<br>
                                <strong>Preview:</strong> {source['chunk_preview']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Stats
                    st.caption(f"Used {result.get('num_chunks_used', 0)} relevant chunks")
                    
                else:
                    st.error(f"❌ {result.get('answer', 'Error processing question') if result else 'Unknown error'}")
    
    # Sample questions
    if not st.session_state.chat_history:
        st.header("💡 Sample Questions")
        st.markdown("""
        Try asking questions like:
        - "What are the main topics covered in the documents?"
        - "Can you summarize the key points?"
        - "What does the document say about [specific topic]?"
        - "Are there any important dates or numbers mentioned?"
        """)

if __name__ == "__main__":
    main()
