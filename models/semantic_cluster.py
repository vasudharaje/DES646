# models/semantic_cluster.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

CLIP_MODEL_NAME = "clip-ViT-B-32"

def embed_images(image_paths, batch_size=8, normalize=True):
    """
    Returns a (N, D) numpy array of CLIP image embeddings.
    Expects image_paths: list[str] of file paths.
    """
    model = SentenceTransformer(CLIP_MODEL_NAME)
    # IMPORTANT: pass PIL Images, not numpy arrays
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    embeddings = model.encode(
        pil_images,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize
    )
    return embeddings

def cluster_embeddings(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels

def organize_by_cluster(image_paths, preds):
    clusters = {}
    for path, c in zip(image_paths, preds):
        clusters.setdefault(int(c), []).append(path)
    return clusters
