# Semantic Search Module - Experiment Report

## 1. Dataset Overview
- **Source**: Local `data/` directory.
- **Documents**: 13 files covering various AI topics (e.g., Computer Vision, NLP, Reinforcement Learning).
- **Format**: Text files (`.txt`).
- **Cleaning**: Standard whitespace normalization during loading.

## 2. Configuration Tested
### Embedding Models
The following Hugging Face models were evaluated:
1. **all-MiniLM-L6-v2** (384 dimensions) - Optimized for speed.
2. **all-mpnet-base-v2** (768 dimensions) - Balanced performance.
3. **paraphrase-MiniLM-L3-v2** (384 dimensions) - Ultra-fast.

### Vector Databases
- **FAISS**: Local file-based index.
- **Chroma**: Persistent database storage.

## 3. Retrieval Performance Verification
A backend verification script (`verify_backend.py`) was executed to validatethe semantic search pipeline.

### Test Query
*Query*: "What is reinforcement learning?"

### Observations
- **Relevance**: All models successfully retrieved the most relevant document (`reinforcement_learning.txt`) as the top result.
- **Scores**:
    - **all-mpnet-base-v2** generally provided higher confidence scores for the top match compared to MiniLM models, reflecting its deeper semantic understanding.
    - **FAISS** and **Chroma** returned identical ranking for the same embeddings, confirming consistency across vector stores.
- **Speed**:
    - building the index with **paraphrase-MiniLM-L3-v2** was noticeably faster.
    - **all-mpnet-base-v2** took longest for embedding generation but offered slightly better separation between relevant and non-relevant documents.

## 4. Conclusion
The Semantic Search Module successfully implements:
- Dynamic dataset loading.
- Configurable embedding models.
- Support for multiple vector databases.
- Accurate semantic retrieval.

All requirements for Phase 1 have been met.
