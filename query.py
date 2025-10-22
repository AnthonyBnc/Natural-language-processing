import argparse, os, numpy as np, hnswlib
from utils import ClipEmbedder, load_meta

def search(query: str, data_dir: str, model_name: str = "clip-ViT-B-32", top_k: int = 5, ef: int = 100):
    index_path = os.path.join(data_dir, "index_hnsw.bin")
    meta_path  = os.path.join(data_dir, "meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise SystemExit("Index not found. Run index_images.py first.")

    meta = load_meta(meta_path)
    embedder = ClipEmbedder(model_name)
    qvec = embedder.embed_text([query]).astype(np.float32)
    dim  = qvec.shape[1]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(index_path)
    index.set_ef(max(ef, top_k))
    labels, distances = index.knn_query(qvec, k=top_k)

    hits = []
    for rank, (i, d) in enumerate(zip(labels[0], distances[0])):
        hits.append({"rank": rank+1, "score": float(1 - d), "path": meta[int(i)]["path"]})
    return hits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model",    default="clip-ViT-B-32")
    ap.add_argument("--top_k",    type=int, default=5)
    ap.add_argument("--ef",       type=int, default=100)
    args = ap.parse_args()

    for r in search(args.q, args.data_dir, args.model, args.top_k, args.ef):
        print(f"#{r['rank']:02d}  score={r['score']:.3f}  {r['path']}")
