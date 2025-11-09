# models/semantic_cluster.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
from sklearn.cluster import KMeans
import torchvision.transforms as T

clip_model = SentenceTransformer('clip-ViT-B-32')  # moderate size

def embed_images(image_paths, batch_size=8):
    imgs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        imgs.append(np.array(im.resize((224,224))).astype(np.uint8))
    embeddings = clip_model.encode(imgs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    return embeddings

def cluster_embeddings(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    preds = kmeans.fit_predict(embeddings.cpu().numpy())
    return preds

def organize_by_cluster(image_paths, preds):
    clusters = {}
    for path, c in zip(image_paths, preds):
        clusters.setdefault(int(c), []).append(path)
    return clusters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    embs = embed_images(args.images)
    preds = cluster_embeddings(embs, n_clusters=args.k)
    clusters = organize_by_cluster(args.images, preds)
    for k,v in clusters.items():
        print("Cluster",k,":",v)
