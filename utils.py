# utils.py
import os, json, numpy as np
from PIL import Image
from typing import List, Dict
from sentence_transformers import SentenceTransformer

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_images(root: str) -> List[str]:
    paths = []
    for base, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                paths.append(os.path.join(base, f))
    return sorted(paths)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

class ClipEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 256
    def embed_text(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    def embed_images(self, pil_images: List[Image.Image]) -> np.ndarray:
        return self.model.encode(pil_images, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

def load_image_safe(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def save_meta(meta_path: str, items: List[Dict]):
    with open(meta_path, "w", encoding="utf-8") as f: json.dump(items, f, indent=2, ensure_ascii=False)

def load_meta(meta_path: str) -> List[Dict]:
    if not os.path.exists(meta_path): return []
    with open(meta_path, "r", encoding="utf-8") as f: return json.load(f)
