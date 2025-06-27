# Advanced RAG AI System

A powerful local Retrieval-Augmented Generation (RAG) system that uses Mistral-7B-Instruct to answer questions based on your uploaded documents. Everything runs locally on your machine - no data is sent to external services.

## 🌟 Features

- **Local LLM**: Uses Mistral-7B-Instruct for high-quality text generation
- **Document Processing**: Supports PDF, DOCX, and TXT files
- **Vector Search**: FAISS or ChromaDB for fast similarity search
- **Web Interface**: Beautiful Streamlit-based UI
- **CLI Support**: Command-line interface for automation
- **Privacy-First**: All processing happens locally
- **Source Attribution**: Shows which documents were used for answers

## 🏗️ Architecture

```
User Uploads Document (PDF/DOCX/TXT)
        ↓
  Document Parser & Chunking
        ↓
     Generate Embeddings
        ↓
   Store in Vector Database (FAISS/ChromaDB)
        ↓
   User Asks Question
        ↓
  Semantic Search for Relevant Chunks
        ↓
 Create Context + Question → Mistral-7B-Instruct
        ↓
       Generate Answer
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended)
- GPU with 8GB+ VRAM (optional but recommended for faster inference)
- 20GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/bonin1/advanced-RAG-AI.git
cd advanced-RAG-AI
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: On first run, the system will download the Mistral-7B-Instruct model (~14GB) and sentence transformer model (~90MB) from Hugging Face. This may take some time depending on your internet connection.

## 🚀 Usage

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run app.py
```

This will open your browser to `http://localhost:8501` where you can:

1. **Initialize the System**: Click "Initialize RAG System" (one-time setup)
2. **Upload Documents**: Drag and drop PDF, DOCX, or TXT files
3. **Ask Questions**: Type questions about your documents
4. **View Sources**: See which document sections were used for answers

### Command Line Interface

For automation or scripting:

```bash
# Add documents to the system
python cli.py --add-docs document1.pdf document2.docx

# Ask a question
python cli.py --query "What are the main findings in the research?"

# View system statistics
python cli.py --stats

# Clear the database
python cli.py --clear
```

## ⚙️ Configuration

Edit `config.py` to customize the system:

```python
# Model settings
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector database
VECTOR_DB_TYPE = "faiss"  # or "chroma"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_CONTEXT = 5

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9
```

## 📁 Project Structure

```
advanced-RAG-AI/
├── app.py                 # Streamlit web interface
├── cli.py                 # Command-line interface
├── config.py              # Configuration settings
├── rag_system.py          # Main RAG orchestrator
├── llm_manager.py         # LLM and embedding models
├── document_processor.py  # Document parsing and chunking
├── vector_database.py     # Vector database implementations
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # Apache 2.0 license
├── uploaded_documents/   # Document storage (created at runtime)
├── vector_db/           # Vector database files (created at runtime)
└── model_cache/         # Downloaded models cache (created at runtime)
```

## 🎯 Usage Examples

### Example Questions

Once you've uploaded documents, try asking:

- "What are the main topics covered in the documents?"
- "Can you summarize the key findings?"
- "What does the document say about [specific topic]?"
- "Are there any important dates or statistics mentioned?"
- "What are the conclusions or recommendations?"

### Sample Workflow

1. **Start the system**:
   ```bash
   streamlit run app.py
   ```

2. **Upload documents**: Research papers, manuals, reports, etc.

3. **Ask questions**: The AI will only answer based on your uploaded content

4. **Review sources**: See exactly which parts of your documents were used

## 🔧 Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce `CHUNK_SIZE` and `MAX_CHUNKS_CONTEXT` in `config.py`
- Close other applications to free up RAM
- Consider using a smaller model (modify `MODEL_NAME`)

**Slow Performance**:
- Ensure you have a GPU with CUDA support
- Reduce `MAX_NEW_TOKENS` for faster responses
- Use FAISS instead of ChromaDB for better performance

**Model Download Issues**:
- Check your internet connection
- Ensure you have enough disk space (20GB+)
- Try clearing the `model_cache/` directory and restarting

**Document Processing Errors**:
- Ensure your documents are not password-protected
- Check file formats are supported (PDF, DOCX, TXT)
- Try processing documents one at a time

### Performance Optimization

**For CPU-only systems**:
```python
# In config.py, use a smaller model
MODEL_NAME = "microsoft/DialoGPT-medium"  # Smaller alternative
```

**For GPU systems**:
- Install CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Increase batch sizes for faster processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the Mistral-7B-Instruct model
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- [Streamlit](https://streamlit.io/) for the web interface framework
- [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search
- [ChromaDB](https://www.trychroma.com/) for vector database functionality

## 📞 Support

For questions, issues, or contributions, please:

1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Provide detailed error messages and system information

---

**Built with ❤️ for local AI and privacy**
