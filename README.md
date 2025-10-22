Text-to-Image Vector Search (CLIP + FAISS + Streamlit)

A minimal end‑to‑end program that indexes a folder of images, stores image embeddings in a FAISS vector database, and lets users type text to retrieve the most relevant images using OpenAI CLIP (via sentence-transformers).

📁 Project structure
vector-search/
├─ images/                 # put your .jpg/.png here
├─ data/
│  ├─ index.faiss          # auto-created FAISS index
│  └─ meta.json            # auto-created metadata for images
├─ app.py                  # Streamlit UI (text → image search)
├─ index_images.py         # build/update the FAISS index from images/
├─ query.py                # CLI querying (text → top-K image paths)
├─ utils.py                # shared utils (embeddings, IO)
├─ requirements.txt
└─ README.md
🚀 Quick start

Create env & install

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Add images into images/ (any nested subfolders are fine).

Index images

python index_images.py --images_dir images --data_dir data --model clip-ViT-B-32

Run the UI

streamlit run app.py

Then open the local URL it prints. Type a prompt (e.g., "a red sports car on a track") and you’ll see the top matches.

Tip: Re-run index_images.py whenever you add/remove images; it intelligently updates.

🧠 How it works (in short)

We use CLIP (e.g., clip-ViT-B-32) to encode text and images into the same vector space.

We store image vectors in FAISS for fast nearest-neighbor search.

At query time, we encode the user text, search FAISS for nearest vectors, and display the corresponding images.

⚙️ Configuration

Change the model via --model (any sentence-transformers model that supports images & text, e.g., clip-ViT-L-14).

Set --top_k in query.py or adjust the Streamlit slider.