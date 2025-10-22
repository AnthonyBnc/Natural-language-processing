import os, streamlit as st, hnswlib
from PIL import Image
import numpy as np
from utils import ClipEmbedder, load_meta

st.set_page_config(page_title="Text → Image Vector Search", layout="wide")

DATA_DIR   = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index_hnsw.bin")
META_PATH  = os.path.join(DATA_DIR, "meta.json")
MODEL_NAME = st.sidebar.selectbox("Embedding model", ["clip-ViT-B-32", "clip-ViT-L-14"], index=0)

st.title("🔎 Text → Image Vector Search (macOS)")

if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
    st.warning("No index found. Please run `python index_images.py` first.")
    st.stop()

meta     = load_meta(META_PATH)
embedder = ClipEmbedder(MODEL_NAME)

with st.form("search_form"):
    q      = st.text_input("Describe what you want to see:", value="a cute cat with blue eyes")
    top_k  = st.slider("Top K", min_value=1, max_value=20, value=6)
    ef     = st.slider("ef (recall/latency)", min_value=20, max_value=400, value=100, help="Higher = better recall, slower")
    submit = st.form_submit_button("Search")

if submit and q.strip():
    qvec = embedder.embed_text([q]).astype(np.float32)
    dim  = qvec.shape[1]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(INDEX_PATH)
    index.set_ef(max(ef, top_k))

    labels, distances = index.knn_query(qvec, k=top_k)
    cols = st.columns(min(top_k, 4))
    for rank, (i, d) in enumerate(zip(labels[0], distances[0])):
        path = meta[int(i)]["path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            st.error(f"Failed to load {path}: {e}")
            continue
        with cols[rank % len(cols)]:
            st.image(img, caption=f"#{rank+1} • {os.path.basename(path)}\nscore={1 - d:.3f}")
