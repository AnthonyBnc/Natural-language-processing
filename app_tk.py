import os, glob, math, tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

# Minimal CLIP wrapper (sentence-transformers)
from sentence_transformers import SentenceTransformer

# ---- Settings ----
IMAGES_DIR = "images"               # folder with your images
MODEL_NAME = "clip-ViT-B-32"        # you can switch to "clip-ViT-L-14"
THUMB_W, THUMB_H = 256, 256         # thumbnail size for display
COLUMNS = 3                         # images per row
DEFAULT_TOPK = 9

SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_images(root: str):
    paths = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(SUPPORTED):
                paths.append(os.path.join(base, f))
    paths.sort()
    return paths

def load_image_rgb(path: str):
    return Image.open(path).convert("RGB")

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    # L2-normalize rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat / norms).astype(np.float32)

class ClipEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 256

    def embed_text(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    def embed_images(self, pil_images):
        return self.model.encode(pil_images, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Text → Image Vector Search (simple macOS)")
        self.geometry("980x720")
        self.minsize(760, 520)

        # UI top controls
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Query:").pack(side=tk.LEFT)
        self.q_var = tk.StringVar(value="a cute cat with blue eyes")
        entry = ttk.Entry(top, textvariable=self.q_var, width=60)
        entry.pack(side=tk.LEFT, padx=6)
        entry.bind("<Return>", lambda e: self.search())

        ttk.Label(top, text="Top K:").pack(side=tk.LEFT, padx=(12, 4))
        self.k_var = tk.IntVar(value=DEFAULT_TOPK)
        ttk.Spinbox(top, from_=1, to=50, textvariable=self.k_var, width=5).pack(side=tk.LEFT)

        ttk.Button(top, text="Search", command=self.search).pack(side=tk.LEFT, padx=10)

        # Scrollable canvas for results
        self.canvas = tk.Canvas(self, borderwidth=0, background="#111111")
        self.scroll_y = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.grid_frame = ttk.Frame(self.canvas)
        self.frame_id = self.canvas.create_window((0,0), window=self.grid_frame, anchor="nw")

        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.grid_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._photos = []  # keep references to PhotoImage objects

        # Load model + images + image embeddings once at startup
        self._init_engine()
        self.search()  # initial search

    def _on_frame_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.frame_id, width=event.width)

    def _init_engine(self):
        # Check images
        self.img_paths = list_images(IMAGES_DIR)
        if not self.img_paths:
            messagebox.showerror("No images", f"No images found in '{IMAGES_DIR}'.")
            self.destroy()
            return

        # Load model
        self.embedder = ClipEmbedder(MODEL_NAME)

        # Embed all images (once)
        pil_batch = []
        img_vecs = []
        B = 32
        for i in range(0, len(self.img_paths), B):
            pil_batch = [load_image_rgb(p) for p in self.img_paths[i:i+B]]
            vec = self.embedder.embed_images(pil_batch)  # normalized already
            img_vecs.append(vec)
        self.img_mat = np.vstack(img_vecs).astype(np.float32)  # (N, D)

        # (Optional) ensure normalized
        self.img_mat = normalize_rows(self.img_mat)

    def search(self):
        q = self.q_var.get().strip()
        if not q:
            return
        k = max(1, int(self.k_var.get()))

        # embed text
        qvec = self.embedder.embed_text([q]).astype(np.float32)  # (1, D)
        qvec = normalize_rows(qvec)[0]  # (D,)

        # cosine similarity (dot since both normalized)
        scores = (self.img_mat @ qvec).ravel()  # (N,)
        # top-k
        idx = np.argpartition(-scores, range(min(k, scores.size)))[:k]
        # sort the top-k by score desc
        idx = idx[np.argsort(-scores[idx])]

        # draw results
        for c in self.grid_frame.winfo_children():
            c.destroy()
        self._photos.clear()

        for rank, i in enumerate(idx):
            path = self.img_paths[int(i)]
            score = float(scores[int(i)])
            try:
                img = load_image_rgb(path)
                img.thumbnail((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            except Exception:
                # fallback tile if an image fails
                img = Image.new("RGB", (THUMB_W, THUMB_H), (40, 40, 40))
                photo = ImageTk.PhotoImage(img)

            self._photos.append(photo)  # keep reference
            cell = ttk.Frame(self.grid_frame, padding=6)
            r, c = divmod(rank, COLUMNS)
            cell.grid(row=r, column=c, sticky="nwes")

            ttk.Label(cell, image=photo).pack()
            name = os.path.basename(path)
            ttk.Label(cell, text=f"#{rank+1}  {name}\nsim={score:.3f}").pack()

if __name__ == "__main__":
    App().mainloop()
