# ============================================================
# INCREMENTAL INDEX UPDATER
# Adds ONLY new images to FAISS index
# ============================================================

import os
import sys
import json
import faiss
import numpy as np

from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import (
    DATASET_ROOT,
    FAISS_PATH,
    METADATA_PATH,
    EMBEDDINGS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    VALID_EXTENSIONS,
    BATCH_SIZE
)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n================================================")
print("LOADING CLIP MODEL")
print("================================================")

model = SentenceTransformer(MODEL_PATH)

print("Model Loaded Successfully")

# ============================================================
# LOAD EXISTING INDEX
# ============================================================

print("\n================================================")
print("LOADING EXISTING DATABASE")
print("================================================")

index = faiss.read_index(FAISS_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

old_embeddings = np.load(EMBEDDINGS_PATH)

print("Indexed Images:", len(metadata))

if len(metadata) != index.ntotal or len(metadata) != old_embeddings.shape[0]:
    print("\nWARNING: metadata.json, embeddings.npy, and FAISS index are OUT OF SYNC.")
    print(f"  metadata entries : {len(metadata)}")
    print(f"  FAISS vectors    : {index.ntotal}")
    print(f"  embeddings rows  : {old_embeddings.shape[0]}")
    print("Run extract_embeddings.py to resync embeddings.npy with the FAISS index,")
    print("or rebuild from scratch with build_image_index.py before continuing.")
    sys.exit(1)

# ============================================================
# FIND EXISTING PATHS
# ============================================================

indexed_paths = set()

for item in metadata:
    indexed_paths.add(item["path"])

# ============================================================
# SCAN DATASET
# ============================================================

print("\n================================================")
print("SCANNING DATASET")
print("================================================")

new_images = []

for root, _, files in os.walk(DATASET_ROOT):

    for file in files:

        ext = os.path.splitext(file)[1].lower()

        if ext not in VALID_EXTENSIONS:
            continue

        full_path = os.path.join(root, file)
        relative_path = os.path.relpath(full_path, PROJECT_ROOT)

        if relative_path not in indexed_paths:
            new_images.append(full_path)

print("New Images Found:", len(new_images))

# ============================================================
# EXIT IF NOTHING NEW
# ============================================================

if len(new_images) == 0:
    print("\nNo new images detected.")
    sys.exit(0)

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\n================================================")
print("GENERATING NEW EMBEDDINGS")
print("================================================")

new_embeddings = []

next_id = len(metadata)

for i in tqdm(range(0, len(new_images), BATCH_SIZE)):

    batch_paths = new_images[i:i + BATCH_SIZE]

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
        batch_size=BATCH_SIZE,
        normalize_embeddings=True
    )

    new_embeddings.extend(embeddings)

    for path in valid_paths:
        metadata.append({
            "id": next_id,
            "path": os.path.relpath(path, PROJECT_ROOT),
            "category": os.path.basename(os.path.dirname(path))
        })
        next_id += 1

# ============================================================
# UPDATE FAISS
# ============================================================

print("\n================================================")
print("UPDATING FAISS INDEX")
print("================================================")

new_embeddings = np.array(new_embeddings, dtype=np.float32)

index.add(new_embeddings)

faiss.write_index(index, FAISS_PATH)

# ============================================================
# UPDATE EMBEDDINGS.NPY
# ============================================================

updated_embeddings = np.vstack([old_embeddings, new_embeddings])

np.save(EMBEDDINGS_PATH, updated_embeddings)

# ============================================================
# UPDATE METADATA
# ============================================================

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# ============================================================
# DONE
# ============================================================

print("\n================================================")
print("UPDATE COMPLETE")
print("================================================")

print("New Images Added:", len(new_images))
print("Total Indexed Images:", len(metadata))
print("FAISS Vectors:", index.ntotal)