# Model configuration
MODEL_NAME = "teknium/OpenHermes-2.5-Mistral-7B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector database settings
VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "chroma"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_CONTEXT = 5

# Document upload settings
UPLOAD_DIR = "uploaded_documents"
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# LLM settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9

# Cache settings
CACHE_DIR = "model_cache"
VECTOR_DB_PATH = "vector_db"
