# ============================================================
# TEXT → IMAGE SEARCH ENGINE (FIXED + VISUAL OUTPUT)
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
# LOAD FAISS INDEX
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

print("Metadata Loaded:", len(metadata), "records")

# ============================================================
# LOAD CLIP MODEL
# ============================================================

print("\n================================================")
print("LOADING CLIP MODEL (ViT-L/14)")
print("================================================")

model = SentenceTransformer(MODEL_PATH)

print("Model Loaded Successfully")

# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_images(query, top_k=6):

    query = query.strip()

    if not query:
        print("Empty query. Please enter a valid search term.")
        return []

    print("\nSearching for:", query)

    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        results.append({
            "rank": i + 1,
            "image_path": os.path.join(PROJECT_ROOT, metadata[idx]["path"]),
            "category": metadata[idx]["category"],
            "score": float(distances[0][i])
        })

    return results

# ============================================================
# DISPLAY RESULTS (IMAGE GRID VIEW)
# ============================================================

def show_results(results, query):

    if not results:
        print("No results to display.")
        return

    cols = 3
    rows = (len(results) + cols - 1) // cols

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Top Results for: {query}", fontsize=14)

    for i, r in enumerate(results):
        try:
            img = Image.open(r["image_path"])
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{r['category']}\n{r['score']:.3f}")
        except Exception as e:
            print("Error loading image:", r["image_path"], e)

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":

    print("\n================================================")
    print("TEXT → IMAGE SEARCH READY")
    print("Type 'exit' to stop")
    print("================================================")

    while True:

        query = input("\nEnter search query: ")

        if query.lower() == "exit":
            break

        results = search_images(query, top_k=8)

        if not results:
            continue

        print("\nTop Results:\n")

        for r in results:
            print(f"{r['rank']}. {r['image_path']}")
            print(f"   Category: {r['category']}")
            print(f"   Score: {r['score']:.4f}")
            print()

        show_results(results, query)