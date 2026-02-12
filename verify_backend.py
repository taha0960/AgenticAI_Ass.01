import os
import sys
import sys
import shutil

# Add app to path to handle internal imports in main.py
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from main import load_documents, chunk_documents, get_embedding_model, build_vector_store, search
from config import DATA_DIR, EMBEDDING_MODELS, VECTOR_DB_OPTIONS

def verify_backend():
    print("--- Starting Backend Verification ---")

    # 1. Load Documents
    print(f"\n[1] Loading documents from {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} does not exist.")
        return
    
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents.")
    if len(docs) == 0:
        print("Warning: No documents loaded. Please add .txt files to data/.")
        return

    # 2. Chunk Documents
    print("\n[2] Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Test Embedding Models and Vector Stores
    query = "What is reinforcement learning?"
    print(f"\n[3] Testing Search with query: '{query}'")

    # Iterate over all models
    for model_label, model_name in EMBEDDING_MODELS.items():
        print(f"\n   -> Using Model: {model_label}")
        try:
            embedding_model = get_embedding_model(model_name)
        except Exception as e:
            print(f"Error loading model {model_label}: {e}")
            continue

        for db_type in VECTOR_DB_OPTIONS:
            print(f"\n      -> Testing Vector DB: {db_type}")
            try:
                # Clean up previous ChromaDB if exists to ensure fresh test
                if db_type == "Chroma":
                    chroma_dir = os.path.join("Vector_Store", "chroma_db")
                    if os.path.exists(chroma_dir):
                        try:
                            shutil.rmtree(chroma_dir)
                        except:
                            pass

                vs = build_vector_store(chunks, embedding_model, db_type)
                results = search(vs, query, top_k=3)
                
                print(f"         Found {len(results)} results.")
                for i, (doc, score) in enumerate(results):
                    src = os.path.basename(doc.metadata.get("source", "unknown"))
                    print(f"         {i+1}. {src} (Score: {score:.4f})")
                    # print(f"            Snippet: {doc.page_content[:50]}...")

            except Exception as e:
                print(f"Error testing {db_type} with {model_label}: {e}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_backend()
