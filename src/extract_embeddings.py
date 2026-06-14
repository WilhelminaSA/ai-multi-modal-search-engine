import faiss
import numpy as np

from config import FAISS_PATH, EMBEDDINGS_PATH

print("\nLoading FAISS Index...")

index = faiss.read_index(FAISS_PATH)

print("Vectors:", index.ntotal)

embeddings = index.reconstruct_n(0, index.ntotal)

np.save(EMBEDDINGS_PATH, embeddings)

print("\nSaved:")
print(EMBEDDINGS_PATH)

print("\nShape:", embeddings.shape)