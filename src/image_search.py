# ============================================================
# IMAGE → IMAGE SEARCH ENGINE
# AI MULTIMODAL SEARCH ENGINE
# ============================================================

import os
import json
import faiss
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from sentence_transformers import SentenceTransformer

from config import FAISS_PATH, METADATA_PATH, MODEL_PATH, PROJECT_ROOT

# ============================================================
# LOAD FAISS
# ============================================================

print("\n================================================")
print("LOADING FAISS INDEX")
print("================================================")

index = faiss.read_index(FAISS_PATH)

print("FAISS Index Loaded")
print("Total Vectors:", index.ntotal)

# ============================================================
# LOAD METADATA
# ============================================================

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Metadata Loaded:", len(metadata))

# ============================================================
# LOAD MODEL
# ============================================================

print("\n================================================")
print("LOADING CLIP MODEL")
print("================================================")

model = SentenceTransformer(MODEL_PATH)

print("Model Loaded Successfully")

# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_similar_images(image_path, top_k=10):

    image = Image.open(image_path).convert("RGB")

    query_embedding = model.encode(image, normalize_embeddings=True)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        results.append({
            "rank": rank + 1,
            "path": os.path.join(PROJECT_ROOT, metadata[idx]["path"]),
            "category": metadata[idx]["category"],
            "score": float(distances[0][rank])
        })

    return results

# ============================================================
# DISPLAY RESULTS
# ============================================================

def show_results(query_image_path, results):

    total_images = len(results) + 1
    cols = 3
    rows = (total_images + cols - 1) // cols

    plt.figure(figsize=(12, 8))

    query_image = Image.open(query_image_path)
    plt.subplot(rows, cols, 1)
    plt.imshow(query_image)
    plt.axis("off")
    plt.title("QUERY IMAGE")

    for i, result in enumerate(results):
        try:
            img = Image.open(result["path"])
            plt.subplot(rows, cols, i + 2)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{result['category']}\n{result['score']:.3f}")
        except Exception as e:
            print("Error:", result["path"])
            print(e)

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n================================================")
    print("IMAGE → IMAGE SEARCH READY")
    print("Type 'exit' to stop")
    print("================================================")

    while True:

        image_path = input("\nEnter image path: ").strip()

        if image_path.lower() == "exit":
            break

        if not os.path.exists(image_path):
            print("\nImage not found.")
            continue

        print("\nSearching...")

        results = search_similar_images(image_path, top_k=6)

        print("\nTop Results:\n")

        for result in results:
            print(f"{result['rank']}. {result['path']}")
            print(f"   Category: {result['category']}")
            print(f"   Score: {result['score']:.4f}")
            print()

        show_results(image_path, results)