import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

from config import (
    VECTOR_STORE_DIR,
    SUPPORTED_EXTENSIONS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ── Document Loading ────────────────────────────────────────────

def load_documents(directory: str) -> list:
    """Load all supported documents (.txt, .pdf) from the given directory."""
    docs = []
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.join(directory, fname)
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(fpath)
            else:
                loader = TextLoader(fpath, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
    return docs


def get_dataset_stats(directory: str) -> dict:
    """Return basic statistics about the dataset in *directory*."""
    files = [
        f
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    total_size = sum(
        os.path.getsize(os.path.join(directory, f)) for f in files
    )
    return {
        "num_files": len(files),
        "filenames": files,
        "total_size_kb": round(total_size / 1024, 2),
    }


# ── Text Chunking ───────────────────────────────────────────────

def chunk_documents(docs: list) -> list:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# ── Embedding Model ─────────────────────────────────────────────

def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings wrapper for *model_name*."""
    return HuggingFaceEmbeddings(model_name=model_name)


# ── Vector Store ─────────────────────────────────────────────────

def build_vector_store(chunks: list, embedding_model, db_type: str):
    """Create a vector store from document chunks."""
    if db_type == "FAISS":
        store = FAISS.from_documents(chunks, embedding_model)
        store.save_local(os.path.join(VECTOR_STORE_DIR, "faiss_index"))
        return store
    else:
        persist_dir = os.path.join(VECTOR_STORE_DIR, "chroma_db")
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        store = Chroma.from_documents(
            chunks,
            embedding_model,
            persist_directory=persist_dir,
        )
        return store


# ── Semantic Search ──────────────────────────────────────────────

def _l2_to_similarity(l2_dist: float) -> float:
    """Convert L2 distance to a 0-1 cosine similarity score.

    For unit-normalised embeddings (sentence-transformers default):
        L2^2 = 2 - 2*cos(theta)  =>  cos(theta) = 1 - L2^2 / 2
    We clamp the result to [0, 1].
    """
    sim = 1.0 - (l2_dist ** 2) / 2.0
    return max(0.0, min(1.0, sim))


def search(vector_store, query: str, top_k: int = 5) -> list:
    """Return the top-k most relevant chunks for *query*.

    Each element is a tuple (Document, similarity) where similarity is
    a float in [0, 1] (higher = more relevant).
    """
    raw = vector_store.similarity_search_with_score(query, k=top_k)
    return [(doc, _l2_to_similarity(dist)) for doc, dist in raw]
