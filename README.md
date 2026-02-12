[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/15si9kMD)
# CS-4015 Agentic AI  
## Homework 1 – Phase 1: Semantic Search Module

📄 **Assignment Description:** [HW1_Phase1_AgenticAI.pdf](HW1_Phase1_AgenticAI.pdf)

---

## 📖 Project Overview
This project implements the **memory system** for an AI Research Assistant. It is a **Semantic Search Engine** capable of retrieving academic documents based on meaning rather than keyword matching.

Key technologies:
- **Streamlit**: For the interactive GUI.
- **LangChain**: For orchestration and document processing.
- **Hugging Face**: For state-of-the-art embedding models.
- **FAISS & Chroma**: For efficient vector storage and retrieval.

---

## 🚀 Key Features
1.  **Dynamic Dataset Loading**: Upload PDF/TXT files or use the default `data/` directory.
2.  **Configurable Embeddings**: Choose between multiple models (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`).
3.  **Vector Store Options**: Switch between **FAISS** (in-memory/local) and **Chroma** (persistent).
4.  **Semantic Search**: Query your documents and get relevance-ranked results.
5.  **Backend Verification**: Automated script to verify core functionality.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### Setup Steps
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/taha0960/AgenticAI_Ass.01.git
    cd AgenticAI_Ass.01
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

---

## 💻 Usage

### 1. Run the GUI Application
Launch the Streamlit interface:
```bash
streamlit run app/gui.py
```
- Open your browser at the provided URL (usually `http://localhost:8501`).
- **Sidebar**: Select your dataset, embedding model, and vector database.
- **Build Index**: Click to process documents.
- **Search**: Enter a query to find relevant context.

### 2. Run Verification Script
To verify the backend logic (loading, embedding, searching) without the GUI:
```bash
python verify_backend.py
```
This script acts as a test suite, checking all configured models and databases.

---

## 📁 Project Structure
```
.
├── app/
│   ├── config.py         # Configuration settings (paths, models, constants)
│   ├── gui.py            # Streamlit frontend application
│   └── main.py           # Core logic (loading, chunking, embedding, searching)
├── data/                 # Default dataset directory
├── experiments/
│   └── report/           # Experiment reports and findings
├── Vector_Store/         # Storage for generated vector indices
├── verify_backend.py     # script to verify backend functionality
├── requirement.txt       # Python dependencies
└── README.md             # Project documentation
```

---

## 📦 Deliverables
- **GUI Application**: Fully functional via `app/gui.py`.
- **Source Code**: Complete implementation in `app/`.
- **Report**: Available in `experiments/report/report.md`.

---
**Author**: Taha
