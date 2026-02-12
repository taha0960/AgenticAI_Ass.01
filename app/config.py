import os

# ──────────────────────────────────────────────
# Directory Paths (resolved relative to project root)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "Vector_Store")

# ──────────────────────────────────────────────
# Supported File Types
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".txt", ".pdf"]

# ──────────────────────────────────────────────
# HuggingFace Embedding Models (user selects from this list)
# ──────────────────────────────────────────────
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2 (Fast, 384d)": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (Balanced, 768d)": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-MiniLM-L3-v2 (Fastest, 384d)": "sentence-transformers/paraphrase-MiniLM-L3-v2",
}

# ──────────────────────────────────────────────
# Vector Database Options
# ──────────────────────────────────────────────
VECTOR_DB_OPTIONS = ["FAISS", "Chroma"]

# ──────────────────────────────────────────────
# Text Chunking Configuration
# ──────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ──────────────────────────────────────────────
# Retrieval Defaults
# ──────────────────────────────────────────────
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
