# index_images.py
import argparse, os, numpy as np, hnswlib
from tqdm import tqdm
from utils import find_images, ensure_dir, ClipEmbedder, load_image_safe, save_meta

def build_index(images_dir: str, data_dir: str, model_name: str = "clip-ViT-B-32",
                batch_size: int = 32, M: int = 32, ef_construction: int = 200):
    ensure_dir(data_dir)
    index_path = os.path.join(data_dir, "index_hnsw.bin")
    meta_path  = os.path.join(data_dir, "meta.json")

    embedder = ClipEmbedder(model_name)
    img_paths = find_images(images_dir)
    if not img_paths: raise SystemExit("No images found to index.")

    vectors = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Embedding"):
        batch_paths = img_paths[i:i+batch_size]
        imgs = [load_image_safe(p) for p in batch_paths]
        vec  = embedder.embed_images(imgs)  # normalized
        vectors.append(vec)

    mat = np.vstack(vectors).astype(np.float32)
    dim = mat.shape[1]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=mat.shape[0], ef_construction=ef_construction, M=M)
    index.add_items(mat, np.arange(mat.shape[0]))
    index.set_ef(50)
    index.save_index(index_path)

    meta = [{"path": p, "mtime": os.path.getmtime(p)} for p in img_paths]
    save_meta(meta_path, meta)
    print(f"Index built: {index_path} (vectors: {mat.shape[0]}, dim: {dim})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="images")
    ap.add_argument("--data_dir",   default="data")
    ap.add_argument("--model",      default="clip-ViT-B-32")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef_construction", type=int, default=200)
    args = ap.parse_args()
    build_index(args.images_dir, args.data_dir, args.model, args.batch_size, args.M, args.ef_construction)
