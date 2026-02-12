import os
import sys
import tempfile
import shutil
import streamlit as st

# Ensure app/ is on the path so local imports work when launched via
# ``streamlit run app/gui.py`` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR,
    EMBEDDING_MODELS,
    VECTOR_DB_OPTIONS,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    SUPPORTED_EXTENSIONS,
)
from main import (
    load_documents,
    get_dataset_stats,
    chunk_documents,
    get_embedding_model,
    build_vector_store,
    search,
)

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="Semantic Search Engine", layout="wide")
st.title("Semantic Search Engine")
st.caption("AI Research Assistant  –  Memory Module  |  HW1 Phase 1")

# ── Sidebar: Dataset Selection ───────────────────────────────────
st.sidebar.header("1. Dataset")

upload_mode = st.sidebar.radio(
    "How would you like to provide documents?",
    ["Upload files", "Use existing data/ folder"],
)

dataset_dir = DATA_DIR  # default

if upload_mode == "Upload files":
    uploaded = st.sidebar.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        tmp_dir = os.path.join(tempfile.gettempdir(), "semantic_search_uploads")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        for f in uploaded:
            with open(os.path.join(tmp_dir, f.name), "wb") as out:
                out.write(f.getbuffer())
        dataset_dir = tmp_dir
        st.sidebar.success(f"{len(uploaded)} file(s) uploaded.")
    else:
        st.sidebar.info("Please upload at least one document.")
else:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    dir_files = [
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if dir_files:
        st.sidebar.success(f"Found {len(dir_files)} file(s) in data/ folder.")
    else:
        st.sidebar.warning(
            "No supported files in data/ folder. Add .txt or .pdf files."
        )

# Show dataset statistics
if os.path.isdir(dataset_dir):
    stats = get_dataset_stats(dataset_dir)
    if stats["num_files"] > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dataset Stats")
        st.sidebar.write(f"**Documents:** {stats['num_files']}")
        st.sidebar.write(f"**Total size:** {stats['total_size_kb']} KB")
        with st.sidebar.expander("File list"):
            for fn in stats["filenames"]:
                st.write(f"- {fn}")

# ── Sidebar: Model & Vector DB ───────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("2. Configuration")

model_label = st.sidebar.selectbox(
    "Embedding Model",
    list(EMBEDDING_MODELS.keys()),
)
model_name = EMBEDDING_MODELS[model_label]

db_type = st.sidebar.selectbox("Vector Database", VECTOR_DB_OPTIONS)

# ── Build Index Button ────────────────────────────────────────────
st.sidebar.markdown("---")
build_clicked = st.sidebar.button(
    "Build Index", type="primary", use_container_width=True
)

if build_clicked:
    stats = get_dataset_stats(dataset_dir)
    if stats["num_files"] == 0:
        st.error("No documents found. Upload or add files first.")
    else:
        with st.spinner("Loading documents..."):
            docs = load_documents(dataset_dir)
        st.info(f"Loaded {len(docs)} document page(s).")

        with st.spinner("Chunking documents..."):
            chunks = chunk_documents(docs)
        st.info(f"Created {len(chunks)} chunks.")

        with st.spinner(f"Generating embeddings with **{model_label}**..."):
            emb_model = get_embedding_model(model_name)

        with st.spinner(f"Building **{db_type}** vector store..."):
            vs = build_vector_store(chunks, emb_model, db_type)

        # Persist in session state
        st.session_state["vector_store"] = vs
        st.session_state["emb_model_label"] = model_label
        st.session_state["db_type"] = db_type
        st.session_state["num_chunks"] = len(chunks)
        st.success("Index built successfully!")

# ── Main Area: Semantic Search ────────────────────────────────────
st.markdown("---")
st.header("Semantic Search")

if "vector_store" not in st.session_state:
    st.info("Build an index first using the sidebar controls.")
else:
    st.write(
        f"**Active index:** {st.session_state['emb_model_label']} / "
        f"{st.session_state['db_type']} "
        f"({st.session_state['num_chunks']} chunks)"
    )

    query = st.text_input("Enter your search query:")
    top_k = st.slider(
        "Top-K results", min_value=1, max_value=MAX_TOP_K, value=DEFAULT_TOP_K
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = search(st.session_state["vector_store"], query, top_k)

        if not results:
            st.warning("No results found.")
        else:
            # Sort by similarity descending (highest = most relevant)
            results = sorted(results, key=lambda x: x[1], reverse=True)

            st.subheader(f"Top {len(results)} Results")
            for rank, (doc, sim) in enumerate(results, 1):
                pct = sim * 100
                src = os.path.basename(doc.metadata.get("source", "N/A"))
                with st.expander(
                    f"#{rank}  |  Similarity: {pct:.1f}%  |  {src}",
                    expanded=(rank == 1),
                ):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Source:** `{src}`")
                        if "page" in doc.metadata:
                            st.markdown(f"**Page:** {doc.metadata['page']}")
                    with col2:
                        st.metric("Cosine Similarity", f"{pct:.1f}%")
                    st.markdown("---")
                    st.markdown(doc.page_content)
