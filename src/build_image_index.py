# ============================================================
# CLEAN IMAGE INDEX BUILDER (CLIP ViT-L/14)
# ============================================================

import os
import json
import faiss
import numpy as np

from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import (
    DATASET_ROOT,
    DATABASE_ROOT,
    FAISS_PATH,
    METADATA_PATH,
    EMBEDDINGS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    VALID_EXTENSIONS,
    BATCH_SIZE
)

# ============================================================
# LOAD CLIP MODEL
# ============================================================

print("\n================================================")
print("LOADING CLIP MODEL (ViT-L/14)")
print("================================================")

model = SentenceTransformer(MODEL_PATH)

print("Model Loaded Successfully")

# ============================================================
# SCAN DATASET
# ============================================================

print("\n================================================")
print("SCANNING DATASET")
print("================================================")

image_paths = []

for root, _, files in os.walk(DATASET_ROOT):
    for file in files:
        if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS:
            image_paths.append(os.path.join(root, file))

print(f"Total Images Found: {len(image_paths)}")

# ============================================================
# STORAGE
# ============================================================

all_embeddings = []
metadata = []
image_id = 0

# ============================================================
# PROCESS IN BATCHES
# ============================================================

print("\n================================================")
print("GENERATING EMBEDDINGS")
print("================================================")

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):

    batch_paths = image_paths[i:i + BATCH_SIZE]

    images = []
    valid_paths = []

    for path in batch_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipped (unreadable): {path} -> {e}")
            continue

    if len(images) == 0:
        continue

    embeddings = model.encode(
        images,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE
    )

    all_embeddings.extend(embeddings)

    for path in valid_paths:
        metadata.append({
            "id": image_id,
            "path": os.path.relpath(path, PROJECT_ROOT),
            "category": os.path.basename(os.path.dirname(path))
        })
        image_id += 1

# ============================================================
# FAISS INDEX
# ============================================================

print("\n================================================")
print("BUILDING FAISS INDEX")
print("================================================")

embeddings_array = np.array(all_embeddings, dtype=np.float32)

dim = embeddings_array.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings_array)

# ============================================================
# SAVE FILES
# ============================================================

faiss.write_index(index, FAISS_PATH)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# Save embeddings.npy directly here (no need for extract_embeddings.py step)
np.save(EMBEDDINGS_PATH, embeddings_array)

print("\nDONE")
print("Images Indexed:", len(metadata))
print("Embeddings saved to:", EMBEDDINGS_PATH)